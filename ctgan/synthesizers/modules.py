import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from ctgan.synthesizers.transformer import Encoder


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10, model_type='mlp', 
                 enc_dim=128, n_head=4, n_layers=2, drop_prob=0.5):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim

        if model_type == 'mlp':
            seq = []
            for item in list(discriminator_dim):
                seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
                dim = item
            seq += [Linear(dim, 1)]
            self.dis = Sequential(*seq)
        
        elif model_type == 'transformer':
            self.dis = Encoder(d_model=enc_dim,
                                ffn_hidden=enc_dim*2,
                                n_head=n_head,
                                n_layers=n_layers,
                                drop_prob=drop_prob)
    
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.dis(input_.view(-1, self.pacdim))


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
                                drop_prob=drop_prob)
    
    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.gen(input_)
        return data
