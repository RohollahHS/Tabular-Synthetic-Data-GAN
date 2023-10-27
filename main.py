import torch
import os
import argparse
from ctgan import CTGAN
from utils.data_processing import preprocessing, post_processing
from data_transformer_evaluation import start_evaluation
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('Tabular Synthetic Data', add_help=False)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_size", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument("--hidden_size", type=int, default=256, help='hidden_size for MLP in Generator and Discriminator')
    parser.add_argument('--display_intervals', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=784, 
                        help='input_size for discriminator and output size for generator')
    parser.add_argument('--model_type', type=str, 
                        help="model archeticture in Geneartor and Discriminator", 
                        default='mlp', choices=['mlp','transformer'])

    parser.add_argument('--one_hot_smoothing', type=bool, default=True, choices=[True, False])
    parser.add_argument('--uniform_gama', type=float, default=0.2)
    parser.add_argument('--file_name', type=str, help='dataset name', default='tickets')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data')
    parser.add_argument('--output_path', default='data', type=str, metavar='PATH')
    parser.add_argument('--all_data', type=str, default=False, choices=[True, False])
    parser.add_argument('--resume', help='resume from checkpoint', default=True, choices=[True, False])
    parser.add_argument("--model_name", type=str, default="Debugging")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(f'{args.output_path}')

    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    return args


if __name__ == '__main__':
    args = parse_option()

    # transforming features
    data = preprocessing(args.file_name, args.data_path)

    if args.all_data == False:
        np.random.seed(10)
        idx = np.random.randint(0, data.shape[0], 400)
        data = data.iloc[idx]

    # Names of the columns that are discrete
    discrete_columns = ['task_type','customer_satisfaction', 'creation_date_year',
                        'customer_problem_resolved','user_actioned','user_team']

    ctgan = CTGAN(epochs=args.n_epochs, batch_size=args.batch_size,
                  model_type=args.model_type, args=args)
    ctgan.fit(data, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)

    # save raw synthetic data and post processed synthetic data
    post_processing(synthetic_data, args.file_name, args.data_path)

    start_evaluation(args.file_name, args.data_path, args.model_name)    
