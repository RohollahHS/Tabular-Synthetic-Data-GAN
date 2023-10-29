import pandas as pd
import argparse
from ctgan import CTGAN
from utils.data_processing import preprocessing, post_processing
from data_transformer_evaluation import start_evaluation
import numpy as np
from ctgan import load_demo
import torch


def parse_option():
    parser = argparse.ArgumentParser('Tabular Synthetic Data', add_help=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_type', type=str, 
                        help="model archeticture in Geneartor and Discriminator", 
                        default='mlp', choices=['mlp','transformer'])
    
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--generator_dim', default=(256, 256))
    parser.add_argument('--discriminator_dim', default=(256, 256))

    parser.add_argument('--file_name', type=str, help='dataset name', default='tickets')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data')
    parser.add_argument('--output_path', default='data', type=str, metavar='PATH')

    parser.add_argument('--all_data', type=bool, default=False, choices=[True, False])

    parser.add_argument('--resume', help='resume from checkpoint', default=False, choices=[True, False])
    parser.add_argument('--display_intervals', default=1)

    parser.add_argument("--conditional", type=bool, default=False, choices=[True, False])
    parser.add_argument("--model_name", type=str, default="Debugging")

    # Generator and Discriminator Args
    # Generator Args
    parser.add_argument('--generator_model_type', default='mlp', choices=['mlp','transformer'])
    parser.add_argument('--generator_d_model', default=128)
    parser.add_argument('--generator_n_head', default=4)
    parser.add_argument('--generator_n_layers', default=2)
    parser.add_argument('--generator_drop_prob', default=0.2)
    # Discriminator Args
    parser.add_argument('--discriminator_model_type', default='mlp', choices=['mlp','transformer'])
    parser.add_argument('--discriminator_d_model', default=128)
    parser.add_argument('--discriminator_n_head', default=4)
    parser.add_argument('--discriminator_n_layers', default=2)
    parser.add_argument('--discriminator_drop_prob', default=0.2)

    args = parser.parse_args()
    
    print('Args:')
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
        data = data.iloc[sorted(idx)]
        # data = data.drop(columns=['delta_creation_date'], axis=1)

    discrete_columns = [
        'task_type',
        'customer_satisfaction',
        'customer_problem_resolved',
        'user_actioned',
        'user_team',]
    
    ctgan = CTGAN(args=args)
    ctgan.fit(data, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(data.shape[0])

    # save raw synthetic data and post processed synthetic data
    post_processing(synthetic_data, args.file_name, args.data_path)

    start_evaluation(args.file_name, args.data_path, args.model_name, drop_column=True)
