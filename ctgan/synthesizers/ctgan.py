"""CTGAN module."""

import warnings
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional
from torch.autograd import Variable

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from utils.save_records import save_loss_records, save_plots
from utils.save_load_model import load_checkpoint, save_model
from ctgan.synthesizers.modules import Generator, Discriminator


class CTGAN(BaseSynthesizer):

    def __init__(self, embedding_dim=64, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True, model_type='mlp', args=None):

        assert batch_size % 2 == 0

        self._embedding_dim = args.embedding_dim
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

        num_epochs = self.args.n_epochs

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(self._embedding_dim, 
                                    self._generator_dim, 
                                    data_dim, 
                                    model_type=self.model_type,
                                    args=self.args
                                    ).to(self._device)
        
        discriminator = Discriminator(data_dim, 
                                      self._discriminator_dim, 
                                      model_type=self.model_type, 
                                      args=self.args
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

        optimizerG = optim.Adam(self._generator.parameters(), lr=0.0002)
        optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002)
        
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

        criterion = torch.nn.BCELoss()

        criterion.to(self._device)

        d_losses = np.zeros(num_epochs)
        g_losses = np.zeros(num_epochs)
        real_scores = np.zeros(num_epochs)
        fake_scores = np.zeros(num_epochs)

        def reset_grad():
            optimizerD.zero_grad()
            optimizerG.zero_grad()

        total_step = len(dataloader)
        for epoch in range(curr_epoch, num_epochs):
            for i, samples in enumerate(dataloader):
                samples = samples.to(torch.float32)
                samples = Variable(samples.to(self._device))

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(samples.shape[0], 1).to(self._device)
                real_labels = Variable(real_labels)
                fake_labels = torch.zeros(samples.shape[0], 1).to(self._device)
                fake_labels = Variable(fake_labels)

                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #

                # Compute BCE_Loss using real samples where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # Second term of the loss is always zero since real_labels == 1
                outputs = discriminator(samples)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs
                
                # Compute BCELoss using fake samples
                # First term of the loss is always zero since fake_labels == 0
                z = torch.randn(samples.shape[0], self._embedding_dim).to(self._device)
                z = Variable(z)
                fake_samples = self._generator(z)
                fake_samples = self._apply_activate(fake_samples)

                outputs = discriminator(fake_samples)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs
                
                # Backprop and optimize
                # If D is trained so well, then don't update
                d_loss = d_loss_real + d_loss_fake
                reset_grad()
                d_loss.backward()
                optimizerD.step()
                # ================================================================== #
                #                        Train the generator                         #
                # ================================================================== #

                # Compute loss with fake samples
                z = torch.randn(samples.shape[0], self._embedding_dim).to(self._device)
                z = Variable(z)
                fake_samples = self._generator(z)
                fake_samples = self._apply_activate(fake_samples)
                outputs = discriminator(fake_samples)
                
                # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
                # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
                g_loss = criterion(outputs, real_labels)
                
                # Backprop and optimize
                # if G is trained so well, then don't update
                reset_grad()
                g_loss.backward()
                optimizerG.step()
                # =================================================================== #
                #                          Update Statistics                          #
                # =================================================================== #
                d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
                g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
                real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
                fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))
                
                if (i+1) % self.args.display_intervals == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                        .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                                real_score.mean().item(), fake_score.mean().item()))

            # Save and plot Statistics
            save_plots(d_losses, g_losses, fake_scores, real_scores, 
                       num_epochs, self.args.output_path, self.args.model_name)

            generator_loss_list.append(g_losses.tolist())
            discriminator_loss_list.append(d_losses.tolist())

            save_model(self._generator,
                       discriminator,
                       optimizerG,
                       optimizerD,
                       epoch,
                       generator_loss_list,
                       discriminator_loss_list,
                       self.args.output_path,
                       self.args.model_name)
            
    @random_state
    def sample(self, n):
        steps = n // self._batch_size + 1
        data = []
        for _ in range(steps):

            z = Variable(torch.randn(self._batch_size, self._embedding_dim).to(self._device))

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
