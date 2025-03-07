import pandas as pd
from typing import Optional


def load_csv_file(file_name: str) -> Optional[pd.DataFrame, None]:

    if not file_name.endswith('.csv'):
        file_name += '.csv'
    try:
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        print('file name does not exist in embeddings/data directory')
    else:
        return data