import argparse
from ctgan import CTGAN
from ctgan import load_demo
from ctgan import post_processing


def parse_option():
    parser = argparse.ArgumentParser('Tabular Synthetic Data', add_help=False)
    # easy config modification
    parser.add_argument('--model_type', type=str, 
                        help="model archeticture in Geneartor and Discriminator", 
                        default='mlp', choices=['mlp','transformer'])
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data/new_tickets')
    parser.add_argument('--all_data', type=str, default='no', choices=['yes, no'])
    parser.add_argument('--resume', help='resume from checkpoint', default=True)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument("--model_name", type=str, default="Debugging")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_option()

    real_data = load_demo(args.data_path)

    if args.all_data == 'no':
        real_data = real_data.iloc[:200]

    # Names of the columns that are discrete
    discrete_columns = [
        'task_type',
        'customer_satisfaction',
        'customer_problem_resolved',
        'user_actioned',
        'user_team',
    ]

    ctgan = CTGAN(epochs=args.n_epochs,
                  batch_size=args.batch_size, 
                  model_type=args.model_type)
    ctgan.fit(real_data, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)
    # synthetic_data.to_csv(f'data/synthetic_{dataset_name}.csv', index=False)

    post_processing(synthetic_data)
