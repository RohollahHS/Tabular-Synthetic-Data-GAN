import argparse
from ctgan import CTGAN
from utils.data_processing import preprocessing, post_processing
from data_transformer_evaluation import start_evaluation
import numpy as np
from ctgan import load_demo


def parse_option():
    parser = argparse.ArgumentParser('Tabular Synthetic Data', add_help=False)

    parser.add_argument('--model_type', type=str, 
                        help="model archeticture in Geneartor and Discriminator", 
                        default='mlp', choices=['mlp','transformer'])
    
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--n_epochs', type=int, default=100)

    parser.add_argument('--file_name', type=str, help='dataset name', default='tickets')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data')
    parser.add_argument('--output_path', default='data', type=str, metavar='PATH')

    parser.add_argument('--all_data', type=str, default=False, choices=[True, False])

    parser.add_argument('--resume', help='resume from checkpoint', default=False, choices=[True, False])
    parser.add_argument('--display_intervals', default=1)

    parser.add_argument("--model_name", type=str, default="Debugging")

    args = parser.parse_args()
    
    print('Args:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    return args


if __name__ == '__main__':
    real_data = load_demo()
    np.random.seed(10)
    idx = np.random.randint(0, real_data.shape[0], 400)
    real_data = real_data.iloc[:400]

    # Names of the columns that are discrete
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    ctgan = CTGAN(epochs=100)
    ctgan.fit(real_data, discrete_columns)


    # args = parse_option()

    # # transforming features
    # data = preprocessing(args.file_name, args.data_path)
    # if args.all_data == False:
    #     np.random.seed(10)
    #     idx = np.random.randint(0, data.shape[0], 400)
    #     data = data.iloc[idx]
    #     # data = data.drop(columns=['delta_creation_date'], axis=1)

    # discrete_columns = [
    #     'task_type',
    #     'customer_satisfaction',
    #     'customer_problem_resolved',
    #     'user_actioned',
    #     'user_team',
    # ]
    # ctgan = CTGAN(epochs=100)
    # ctgan.fit(data, discrete_columns)

    # Create synthetic data
    # synthetic_data = ctgan.sample(1000)

    # transforming features
    # data = preprocessing(args.file_name, args.data_path)

    # if args.all_data == False:
    #     np.random.seed(10)
    #     idx = np.random.randint(0, data.shape[0], 400)
    #     data = data.iloc[:200]

    # # Names of the columns that are discrete
    # discrete_columns = [
    #     'task_type',
    #     'customer_satisfaction',
    #     'customer_problem_resolved',
    #     'user_actioned',
    #     'user_team',
    # ]

    # ctgan = CTGAN(epochs=args.n_epochs, batch_size=args.batch_size, 
    #               model_type=args.model_type, args=args)
    # ctgan.fit(data, discrete_columns)

    # # Create synthetic data
    # synthetic_data = ctgan.sample(1000)

    # # save raw synthetic data and post processed synthetic data
    # post_processing(synthetic_data, args.file_name, args.data_path)

    # start_evaluation(args.file_name, args.data_path, args.model_name, True)    
