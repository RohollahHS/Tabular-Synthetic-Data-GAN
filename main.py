import os
import argparse
from ctgan import CTGAN
from utils.data_processing import preprocessing, post_processing
from data_transformer_evaluation import start_evaluation


def parse_option():
    parser = argparse.ArgumentParser('Tabular Synthetic Data', add_help=False)

    parser.add_argument('--model_type', type=str, 
                        help="model archeticture in Geneartor and Discriminator", 
                        default='mlp', choices=['mlp','transformer'])
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=42)
    parser.add_argument('--dis_hidden_size', type=int, default=256)
    parser.add_argument('--gen_hidden_size', type=int, default=256)

    parser.add_argument('--file_name', type=str, help='dataset name', default='tickets')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data')
    parser.add_argument('--output_path', default='data', type=str, metavar='PATH')

    parser.add_argument('--all_data', type=str, default='no', choices=['yes, no'])

    parser.add_argument('--resume', help='resume from checkpoint', default=False, choices=[True, False])

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

    if args.all_data == 'no':
        data = data.iloc[:200]

    # Names of the columns that are discrete
    discrete_columns = ['task_type','customer_satisfaction',
                        'customer_problem_resolved','user_actioned','user_team']

    ctgan = CTGAN(epochs=args.n_epochs, batch_size=args.batch_size, 
                  model_type=args.model_type, args=args)
    ctgan.fit(data, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)

    # save raw synthetic data and post processed synthetic data
    post_processing(synthetic_data, args.file_name, args.data_path)

    start_evaluation(args.file_name, args.data_path, args.model_name)    
