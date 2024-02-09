import json
import pandas as pd
from pathlib import Path
from src.code_processing import decode_code_string


def load_ipython_item(data_path: Path) -> pd.DataFrame:
    """Load and clean umimeprogramovat ipython item database.

    Args:
        data_path (Path): Path to the folder with the dataset or the ipython item file.

    Returns:
        pd.DataFrame: Loaded and cleaned item database.
    """
    # load
    if data_path.is_dir():
        data_path = data_path / 'umimeprogramovatcz-ipython_item.csv'
    item = pd.read_csv(data_path, sep=';', index_col=0)
    # process
    item = item[['name', 'instructions', 'solution']]
    item['instructions'] = item['instructions'].apply(lambda x: eval(x)[0][1])
    item['solution'] = item['solution'].apply(lambda x: eval(x)[0][1]).apply(decode_code_string)
    item = pd.concat([
        item,
        *[
            pd.DataFrame({'name': f'unknown_{i}', 'solution': 'pass', 'instructions': 'unknown'}, index=[idx])
                for i, idx in enumerate([12, 118, 142, 143, 144, 145, 146])
        ]
    ])
    return item


def load_ipython_log(data_path: Path, linter_messages_path:Path=None) -> pd.DataFrame:
    """Load and clean umimeprogramovat ipython log.

    Args:
        data_path (Path): Path to the folder with the dataset or the ipython item file.
        linter_messages_path (Path, optional): Path to the folder with the already generated linter 
            messages. Use None to get the data without the messages.

    Returns:
        pd.DataFrame: Loaded and cleaned log.
    """
    if data_path.is_dir():
        data_path = data_path / 'umimeprogramovatcz-ipython_log.csv'
    # load
    log = pd.read_csv(data_path, sep=';')
    # process
    log.drop_duplicates(inplace=True)
    log.dropna(inplace=True)
    log['time'] = pd.to_datetime(log['time'])
    log['answer'] = log['answer'].apply(decode_code_string)
    log = log[log['answer'].str.strip().astype(bool)]

    if linter_messages_path is not None:
        new_log = []
        counts = log['user'].value_counts()

        for file_path in (linter_messages_path).glob("edulint_results*.json"):
            start, end = file_path.stem.split('_')[2].split('-')
            start, end = int(start), int(end)
            selected_counts = counts[(counts >= start) & (counts < end)]  # noqa: F841
            log_slice = pd.DataFrame(log.query('user in @selected_counts.index'))
            log_slice['linter_messages'] = [' '.join(alist) for alist in json.load(open(file_path, 'r'))]
            new_log.append(log_slice)

        log = pd.concat(new_log)

    return log