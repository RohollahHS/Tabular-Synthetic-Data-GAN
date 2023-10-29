"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from utils.save_records import save_plots
from modules.modules import Discriminator, Generator



class CTGAN(BaseSynthesizer):
    def __init__(self, generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, discriminator_steps=1,
                 log_frequency=True, verbose=False, pac=10, args=None):

        assert args.batch_size % 2 == 0

        self.args = args

        # self._embedding_dim = args.embedding_dim
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


        mean = torch.zeros(self._batch_size, self.args.embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        dataloader = torch.utils.data.DataLoader(train_data.astype(np.float32), 
                                                 shuffle=True, 
                                                 batch_size=self.args.batch_size)
        total_steps = len(dataloader)
        criterion = torch.nn.BCELoss().to(self.args.device)

        d_losses = np.zeros(epochs)
        g_losses = np.zeros(epochs)
        real_scores = np.zeros(epochs)
        fake_scores = np.zeros(epochs)

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch in epoch_iterator:
            for i, real_samples in enumerate(dataloader):
                real_samples = Variable(real_samples.to(self.args.device))

                num_samples = real_samples.shape[0]
                real_labels = torch.ones(num_samples, 1).to(self.args.device)
                real_labels = Variable(real_labels)
                fake_labels = torch.zeros(num_samples, 1).to(self.args.device)
                fake_labels = Variable(fake_labels)

                fakez = torch.randn(num_samples, self.args.embedding_dim).to(self.args.device)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                y_fake = discriminator(fakeact)
                fake_score = torch.sigmoid(y_fake)
                # d_loss_fake = criterion(fake_score, fake_labels)

                y_real = discriminator(real_samples)
                real_score = torch.sigmoid(y_real)
                # d_loss_fake = criterion(real_score, real_labels)

                pen = discriminator.calc_gradient_penalty(
                    real_samples, fakeact, self._device, self.pac)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                optimizerD.zero_grad(set_to_none=False)
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.randn(num_samples, self.args.embedding_dim).to(self.args.device)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                y_fake = discriminator(fakeact)

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
                    .format(epoch, epochs, i+1, total_steps, loss_d.item(), loss_g.item(), 
                            real_score.mean().item(), fake_score.mean().item()))

            # save_plots(d_losses, g_losses, fake_scores, real_scores, 
            #         i, self.args.output_path, self.args.model_name)


            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self.args.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
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