"""CTGAN module."""

import warnings
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm.auto import tqdm
from torch.autograd import Variable

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.transformer import Encoder
from utils.save_records import save_loss_records
from utils.save_load_model import load_checkpoint, save_model
import torch.nn.functional as F

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, model_type='mlp', 
                 enc_dim=128, n_head=4, n_layers=2, drop_prob=0.5):
        super(Discriminator, self).__init__()

        if model_type == 'mlp':
            seq = []
            for item in list(discriminator_dim):
                seq += [Linear(input_dim, item), LeakyReLU(0.2), Dropout(0.5)]
                input_dim = item
            seq += [Linear(input_dim, 1)]
            self.dis = Sequential(*seq)
        
        elif model_type == 'transformer':
            self.dis = Encoder(d_model=enc_dim,
                                ffn_hidden=enc_dim*2,
                                n_head=n_head,
                                n_layers=n_layers,
                                drop_prob=drop_prob,
                                data_dim=input_dim,
                                type_encoder='discriminator'
                                )
    
    def forward(self, input_):
        return F.sigmoid(self.dis(input_))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, model_type='mlp',
                 enc_dim=128, n_head=4, n_layers=2, drop_prob=0.5):
        super(Generator, self).__init__()
        dim = embedding_dim

        if model_type == 'mlp':
            seq = []
            for item in list(generator_dim):
                seq += [Residual(dim, item)]
                dim += item
            seq.append(Linear(dim, data_dim))
            self.gen = Sequential(*seq)
        
        elif model_type == 'transformer':
            self.gen = Encoder(d_model=enc_dim,
                                ffn_hidden=enc_dim*2,
                                n_head=n_head,
                                n_layers=n_layers,
                                drop_prob=drop_prob,
                                data_dim=data_dim,
                                type_encoder='generator'
                                )
    
    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.gen(input_)
        return data


class CTGAN(BaseSynthesizer):

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True, model_type='mlp', args=None):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        self.model_type = model_type

        self.args = args

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim,
            self._generator_dim,
            data_dim,
            model_type=self.model_type,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim,
            self._discriminator_dim,
            model_type=self.model_type,
        ).to(self._device)

        n = 0
        for p in self._generator.parameters():
            n += p.numel()
        print('Number of parameters for Generator:', n)
        
        n = 0
        for p in discriminator.parameters():
            n += p.numel()
        print('Number of parameters for Discriminator:', n)
        print()

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        if not os.path.exists(self.args.output_path):
            os.mkdir(f'{self.args.output_path}')
        
        generator_file_name = 'generator_loss_records'
        discriminator_file_name = 'discriminator_loss_records'

        if self.args.resume:
            (
            self._generator,
            discriminator,
            optimizerG, 
            optimizerD,
            curr_epoch,
            generator_loss_list,
            discriminator_loss_list,
            ) = load_checkpoint(self.args.output_path, 
                                self.args.model_name, 
                                self._generator,
                                discriminator,
                                optimizerG,
                                optimizerD,
                                self._device)
        
        elif self.args.resume == False:
            generator_loss_list = []
            discriminator_loss_list = []
            curr_epoch = 0
            save_loss_records(self.args.output_path, generator_file_name, model_name=self.args.model_name)
            save_loss_records(self.args.output_path, discriminator_file_name, model_name=self.args.model_name)

        dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)

        Tensor = torch.cuda.FloatTensor if self._device.type == 'cuda' else torch.FloatTensor

        adversarial_loss = torch.nn.BCELoss()

        if self._device.type == 'cuda':
            adversarial_loss.cuda()

        for i in range(curr_epoch, epochs):
            for records in dataloader:
                for n in range(self._discriminator_steps):

                    # Adversarial ground truths
                    valid = Variable(Tensor(records.size(0), 1).fill_(1.0), requires_grad=False)
                    fake = Variable(Tensor(records.size(0), 1).fill_(0.0), requires_grad=False)

                    # Configure input
                    real_records = Variable(records.type(Tensor))

                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizerG.zero_grad()

                    # Sample noise as generator input
                    z = Variable(Tensor(np.random.normal(0, 1, (records.shape[0], self._embedding_dim))))

                    # Generate a batch of images
                    gen_records = self._generator(z)

                    # Loss measures generator's ability to fool the discriminator
                    # dis_gen_output = discriminator(gen_records)
                    g_loss = adversarial_loss(discriminator(gen_records), valid)

                    g_loss.backward()
                    optimizerG.step()


                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizerD.zero_grad()

                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = adversarial_loss(discriminator(real_records), valid)
                    fake_loss = adversarial_loss(discriminator(gen_records.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()
                    optimizerD.step()

            generator_loss = g_loss.detach().cpu()
            discriminator_loss = d_loss.detach().cpu()
            
            save_loss_records(self.args.output_path, generator_file_name, loss=generator_loss, epoch=i+1)
            save_loss_records(self.args.output_path, discriminator_file_name, loss=discriminator_loss, epoch=i+1)

            print(f'Epoch: {i+1}/{epochs} | G Loss: {generator_loss:.3f} | D Loss: {discriminator_loss:.3f}')

            generator_loss_list.append(generator_loss)
            discriminator_loss_list.append(discriminator_loss)

            save_model(self._generator,
                       discriminator,
                       optimizerG,
                       optimizerD,
                       i,
                       generator_loss_list,
                       discriminator_loss_list,
                       self.args.output_path,
                       self.args.model_name)
            

    @random_state
    def sample(self, n):
        Tensor = torch.cuda.FloatTensor if self._device.type == 'cuda' else torch.FloatTensor

        steps = n // self._batch_size + 1
        data = []
        for _ in range(steps):
            z = Variable(Tensor(np.random.normal(0, 1, (self._batch_size, self._embedding_dim))))

            fake = self._generator(z)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
