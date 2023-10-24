"""Demo module."""

import pandas as pd

DEMO_URL = 'http://ctgan-demo.s3.amazonaws.com/census.csv.gz'


def load_demo(file_name):
    """Load the demo."""
    # return pd.read_csv(DEMO_URL, compression='gzip')
    return pd.read_csv(f'{file_name}.csv')
