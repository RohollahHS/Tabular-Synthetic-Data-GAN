"""CTGAN module."""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import functional
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from utils.save_records import save_plots
from modules.modules import Discriminator, Generator
from utils.save_load_model import load_checkpoint, save_model



class CTGAN(BaseSynthesizer):
    def __init__(self, generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, discriminator_steps=1,
                 log_frequency=True, verbose=False, pac=10, args=None):

        assert args.batch_size % 2 == 0

        self.args = args

        self._embedding_dim = args.embedding_dim
        self._generator_dim = args.generator_dim
        self._discriminator_dim = args.discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = args.batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = args.epochs
        self.pac = pac

        self._device = torch.device(args.device)

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
    def fit(self, train_data, discrete_columns=()):
        self._validate_discrete_columns(train_data, discrete_columns)

        epochs = self._epochs

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        self.args.data_dim = data_dim

        if self.args.generator_model_type == 'transformer':
            self.args.embedding_dim = data_dim

        if not self.args.conditional:
            self.args.condvec_dim = 0
        else:
            self.args.condvec_dim = self._data_sampler.dim_cond_vec()
        
        self._generator = Generator(args=self.args).to(self._device)

        discriminator = Discriminator(args=self.args).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        n = 0
        for p in self._generator.parameters():
            n += p.numel()
        print('Number of Parameters for Generator:', n)

        n = 0
        for p in discriminator.parameters():
            n += p.numel()
        print('Number of Parameters for Discriminator:', n)

        if not os.path.exists(self.args.output_path):
            os.mkdir(f'{self.args.output_path}')
        
        if self.args.resume:
            (self._generator, 
             discriminator, 
             optimizerG, 
             optimizerD, 
             curr_epoch, 
             d_losses, 
             g_losses, 
             real_scores, 
             fake_scores) = load_checkpoint(self.args, 
                                            self._generator,
                                            discriminator,
                                            optimizerG,
                                            optimizerD)
        
        elif self.args.resume == False:
            curr_epoch = 0
            d_losses = np.zeros(epochs)
            g_losses = np.zeros(epochs)
            real_scores = np.zeros(epochs)
            fake_scores = np.zeros(epochs)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch in range(curr_epoch, epochs):
            for i in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    if self.args.conditional:
                        condvec = self._data_sampler.sample_condvec(self._batch_size)
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                    else:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    fake_score = torch.sigmoid(y_fake)

                    y_real = discriminator(real_cat)
                    real_score = torch.sigmoid(y_real)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)

                if self.args.conditional:
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                else:
                    c1, m1, col, opt = None, None, None, None

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if self.args.conditional:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                else:
                    cross_entropy = 0

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + loss_d.item()*(1./(i+1.))
                g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + loss_g.item()*(1./(i+1.))
                real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
                fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))
                
                if (i+1) % self.args.display_intervals == 0:
                    print('Epoch [{}/{}], Step [{}/{}], loss_d: {:.4f}, loss_g: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                        .format(epoch, epochs, i+1, steps_per_epoch, loss_d.item(), loss_g.item(), 
                                real_score.mean().item(), fake_score.mean().item()))

            # save_plots(d_losses, g_losses, fake_scores, real_scores, 
            #            epoch, self.args.output_path, self.args.model_name)
            
            # save_model(self._generator,
            #            discriminator,
            #            optimizerG,
            #            optimizerD,
            #            epoch,
            #            self.args.output_path,
            #            self.args.model_name,
            #            d_losses,
            #            g_losses,
            #            real_scores,
            #            fake_scores,)

    @random_state
    def sample(self, n):
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self.args.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if self.args.conditional:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)