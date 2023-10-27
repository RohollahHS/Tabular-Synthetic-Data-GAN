"""CTGAN module."""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from torch.autograd import Variable

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from utils.save_records import save_plots
from utils.save_load_model import save_model, load_checkpoint
from ctgan.synthesizers.modules import Generator, Discriminator

def init_weights(m):
    for k in m.modules():
        if isinstance(k, torch.nn.Linear):
            torch.nn.init.xavier_normal_(k.weight)


class CTGAN(BaseSynthesizer):

    def __init__(self, embedding_dim=64, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True, model_type='mlp', args=None):

        assert batch_size % 2 == 0

        self.args = args
        self._device = torch.device(args.device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1, args=None):
        if args.one_hot_smoothing:
            if logits.shape[1] == 1:
                return functional.sigmoid(logits)
            return functional.softmax(logits, dim=-1)

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
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2, args=self.args)
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

        self._transformer = DataTransformer(args=self.args)
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        data_dim = self._transformer.output_dimensions

        self.args.input_size = data_dim

        # Initialize generator and discriminator
        self.G = Generator(self.args).to(self.args.device)
        D = Discriminator(self.args).to(self.args.device)
        criterion = torch.nn.BCELoss().to(self.args.device)

        # Optimizers
        g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.args.lr)
        d_optimizer = torch.optim.Adam(D.parameters(), lr=self.args.lr)

        if self.args.resume:
            (self.G, 
             D, 
             g_optimizer, 
             d_optimizer, 
             curr_epoch, 
             d_losses, 
             g_losses, 
             real_scores, 
             fake_scores) = load_checkpoint(self.args, 
                                            self.G,
                                            D,
                                            g_optimizer,
                                            d_optimizer)
        
        elif self.args.resume == False:
            init_weights(self.G)
            init_weights(D)
            curr_epoch = 0
            d_losses = np.zeros(num_epochs)
            g_losses = np.zeros(num_epochs)
            real_scores = np.zeros(num_epochs)
            fake_scores = np.zeros(num_epochs)

        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=self.args.batch_size, 
                                                shuffle=True)

        n = 0
        for p in self.G.parameters():
            n += p.numel()
        print('Number of parameters for Generator:', n)
        
        n = 0
        for p in D.parameters():
            n += p.numel()
        print('Number of parameters for Discriminator:', n)
        print()

        self.G.to(self.args.device)
        D.to(self.args.device)
        criterion.to(self.args.device)

        self.G.train()
        D.train()

        device = self.args.device
        num_epochs = self.args.n_epochs
        
        if self.args.model_type == 'transformer':
            self.args.latent_size = data_dim
        elif self.args.mdeol_type == 'mlp':
            self.args.latent_size = self.args.latent_size

        def reset_grad():
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

        total_step = len(data_loader)
        for epoch in range(curr_epoch, num_epochs):
            for i, (samples) in enumerate(data_loader):
                samples = Variable(samples.to(device))

                # Create the labels which are later used as input for the BCE loss
                real_labels = torch.ones(samples.shape[0], 1).to(device)
                real_labels = Variable(real_labels)
                fake_labels = torch.zeros(samples.shape[0], 1).to(device)
                fake_labels = Variable(fake_labels)

                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #

                # Compute BCE_Loss using real samples where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # Second term of the loss is always zero since real_labels == 1
                outputs = D(samples)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs
                
                # Compute BCELoss using fake samples
                # First term of the loss is always zero since fake_labels == 0
                z = torch.randn(samples.shape[0], self.args.latent_size).to(device)
                z = Variable(z)
                fake_samples = self._apply_activate(self.G(z))
                outputs = D(fake_samples)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs
                
                # Backprop and optimize
                # If D is trained so well, then don't update
                d_loss = d_loss_real + d_loss_fake
                reset_grad()
                d_loss.backward()
                d_optimizer.step()
                # ================================================================== #
                #                        Train the generator                         #
                # ================================================================== #

                # Compute loss with fake samples
                z = torch.randn(samples.shape[0], self.args.latent_size).to(device)
                z = Variable(z)
                fake_samples = self._apply_activate(self.G(z))
                outputs = D(fake_samples)
                
                # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
                # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
                g_loss = criterion(outputs, real_labels)
                
                # Backprop and optimize
                # if G is trained so well, then don't update
                reset_grad()
                g_loss.backward()
                g_optimizer.step()
                # =================================================================== #
                #                          Update Statistics                          #
                # =================================================================== #
                d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
                g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
                real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
                fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))
                
                if (i+1) % self.args.display_intervals == 0:
                    print('Epoch [{:3d}/{}], Step [{:4d}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                                real_score.mean().item(), fake_score.mean().item()))

            # Save and plot Statistics
            save_plots(d_losses, g_losses, fake_scores, real_scores, 
                    epoch, self.args.output_path, self.args.model_name)

            save_model(self.G,
                       D,
                       g_optimizer,
                       d_optimizer,
                       epoch,
                       self.args.output_path,
                       self.args.model_name,
                       d_losses,
                       g_losses,
                       real_scores,
                       fake_scores,)
            
    @random_state
    def sample(self, n):
        self.G.eval()

        steps = n // self.args.batch_size + 1
        data = []
        for _ in range(steps):

            z = Variable(torch.randn(self.args.batch_size, self.args.latent_size).to(self.args.device))

            with torch.no_grad():
                fake = self.G(z)
                fakeact = self._apply_activate(fake)
            
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)

        data = data[:n]

        return self._transformer.inverse_transform(data)
