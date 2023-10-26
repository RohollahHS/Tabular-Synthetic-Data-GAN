import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch import nn

from ctgan.synthesizers.transformer import Encoder


# class Discriminator(Module):
#     """Discriminator for the CTGAN."""

#     def __init__(self, input_dim, discriminator_dim, model_type='mlp', 
#                  enc_dim=128, n_head=4, n_layers=2, drop_prob=0.5, args=None):
#         super(Discriminator, self).__init__()
        
#         if model_type == 'mlp':
#             hidden_size = args.dis_hidden_size
#             self.dis = nn.Sequential(
#                        nn.Linear(input_dim, hidden_size),
#                        nn.LeakyReLU(0.2),
#                        nn.Linear(hidden_size, hidden_size),
#                        nn.LeakyReLU(0.2),
#                        nn.Linear(hidden_size, 1),
#                        nn.Sigmoid())
        
#         elif model_type == 'transformer':
#             self.dis = Encoder(d_model=enc_dim,
#                                 ffn_hidden=enc_dim*2,
#                                 n_head=n_head,
#                                 n_layers=n_layers,
#                                 drop_prob=drop_prob,
#                                 data_dim=input_dim,
#                                 type_encoder='discriminator'
#                                 )
    
#     def forward(self, input_):
#         return F.sigmoid(self.dis(input_))


# class Residual(Module):
#     """Residual layer for the CTGAN."""

#     def __init__(self, i, o):
#         super(Residual, self).__init__()
#         self.fc = Linear(i, o)
#         self.bn = BatchNorm1d(o)
#         self.relu = ReLU()

#     def forward(self, input_):
#         """Apply the Residual layer to the `input_`."""
#         out = self.fc(input_)
#         out = self.bn(out)
#         out = self.relu(out)
#         return torch.cat([out, input_], dim=1)


# class Generator(Module):
#     """Generator for the CTGAN."""

#     def __init__(self, embedding_dim, generator_dim, data_dim, model_type='mlp',
#                  enc_dim=128, n_head=4, n_layers=2, drop_prob=0.5, args=None):
#         super(Generator, self).__init__()

#         if model_type == 'mlp':
#             hidden_size = args.gen_hidden_size
#             self.gen = nn.Sequential(
#                             nn.Linear(args.embedding_dim, hidden_size),
#                             nn.ReLU(),
#                             nn.Linear(hidden_size, hidden_size),
#                             nn.ReLU(),
#                             nn.Linear(hidden_size, data_dim))
        
#         elif model_type == 'transformer':
#             self.gen = Encoder(d_model=enc_dim,
#                                 ffn_hidden=enc_dim*2,
#                                 n_head=n_head,
#                                 n_layers=n_layers,
#                                 drop_prob=drop_prob,
#                                 data_dim=data_dim,
#                                 type_encoder='generator'
#                                 )
    
#     def forward(self, input_):
#         """Apply the Generator to the `input_`."""
#         data = self.gen(input_)
#         return data


class Generator(nn.Module):
    def __init__(self,  args):
        super(Generator, self).__init__()
        input_size = args.input_size
        latent_size = args.latent_size
        hidden_size = args.hidden_size

        self.model = nn.Sequential(
                    nn.Linear(latent_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, input_size))

    def forward(self, z):
        x = self.model(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        input_size = args.input_size
        hidden_size = args.hidden_size

        self.model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)

        return validity
