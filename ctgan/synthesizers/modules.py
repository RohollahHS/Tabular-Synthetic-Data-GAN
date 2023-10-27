from torch import nn

from ctgan.synthesizers.transformer import Encoder


class Generator(nn.Module):
    def __init__(self,  args):
        super(Generator, self).__init__()
        
        if args.model_type == 'mlp':
            self.model = nn.Sequential(
                        nn.Linear(args.latent_size, args.hidden_size),
                        nn.ReLU(),
                        nn.Linear(args.hidden_size, args.hidden_size),
                        nn.ReLU(),
                        nn.Linear(args.hidden_size, args.input_size))
        
        elif args.model_type == 'transformer':
            self.model = Encoder(d_model=args.hidden_size,
                                 ffn_hidden=args.hidden_size*2,
                                 n_head=args.n_heads,
                                 n_layers=args.n_encoder_layers,
                                 drop_prob=args.drop_prob,
                                 data_dim=args.input_size,
                                 type_encoder='generator')

    def forward(self, z):
        x = self.model(z)
        return x



class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        if args.model_type == 'mlp':
            self.model = nn.Sequential(
                        nn.Linear(args.input_size, args.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(args.hidden_size, args.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(args.hidden_size, 1),
                        nn.Sigmoid())
        
        elif args.model_type == 'transformer':
            self.model = Encoder(d_model=args.hidden_size,
                                 ffn_hidden=args.hidden_size*2,
                                 n_head=args.n_heads,
                                 n_layers=args.n_encoder_layers,
                                 drop_prob=args.drop_prob,
                                 data_dim=args.input_size,
                                 type_encoder='discriminator')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)

        return validity
