import pandas as pd
from pathlib import Path

from src.code_processing import decode_code_string


def load_item(data_path: Path) -> pd.DataFrame:
    """
    Load and clean umimeprogramovat python item dataset.
    TODO add item 12.
    """
    # load
    item = pd.read_csv(data_path / 'umimeprogramovatcz-ipython_item.csv', sep=';', index_col=0)
    # process
    item = item[['name', 'instructions', 'solution']]
    item['instructions'] = item['instructions'].apply(lambda x: eval(x)[0][1])
    item['solution'] = item['solution'].apply(lambda x: eval(x)[0][1]).apply(decode_code_string)
    return item


def load_log(data_path: Path) -> pd.DataFrame:
    """
    Load and clean umimeprogramovat python log dataset.
    TODO drop rows referring to nonexistent items?
    """
    # load
    log = pd.read_csv(data_path / 'umimeprogramovatcz-ipython_log.csv', sep=';')
    # process
    log.drop_duplicates(inplace=True)
    log.dropna(inplace=True)
    log['time'] = pd.to_datetime(log['time'])
    log.set_index('time', inplace=True)
    log['answer'] = log['answer'].apply(decode_code_string)
    log = log[log['answer'].str.strip().astype(bool)]

    return log