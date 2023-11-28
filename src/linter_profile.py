import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from typing import Optional


def make_task_means(item: pd.DataFrame, log: pd.DataFrame) -> pd.Series:
    # for each task
    means = []
    for task_id in item.index:
        # get history
        history = log['linter_messages'][log['item'] == task_id]
        # make profile and get message means
        if len(history) == 0:
            means.append(np.zeros(log['linter_messages'].iloc[0].shape[0]))
        else:
            means.append(history.mean(axis=0))
    return pd.Series(means, item.index)


def make_task_profiles(item: pd.DataFrame, log: pd.DataFrame, norm='l2') -> pd.Series:
    # for each task
    profiles = []
    for task_id in item.index:
        # get history
        history = log['linter_messages'][log['item'] == task_id]
        # make profile and get message means
        if len(history) == 0:
            profiles.append(np.zeros(log['linter_messages'].iloc[0].shape[0]))
        else:
            profiles.append(task_profile(np.vstack(history), norm=norm))
    return pd.Series(profiles, item.index)


def make_user_profiles(log: pd.DataFrame, task_means: Optional[pd.Series]=None, norm: str='l2', decay: float=1.):
    # for each unique user
    profiles = []
    for user_id in set(log['user']):
        # get user history and sort by timestamp
        user_history = log[log['user'] == user_id].sort_values('time')

        # subtract task means if given
        if task_means is not None:
            user_history['linter_messages'] -= user_history.join(task_means, on='item')[task_means.name]

        # make user profile
        profiles.append(pd.Series(user_profile(user_history['linter_messages'], norm, decay).tolist(), user_history.index))

    return pd.concat(profiles)


def user_profile(user_history: pd.Series, norm: str='l2', decay: float=1.):
    """
    history has to be sorted! TODO
    norm: one of l1, l2, max
    decay: 1 means no decay, 0 means maximum decay
    """
    # history must be sorted
    user_history = np.vstack(user_history)

    # rolling sum
    cum_sum = np.zeros(user_history.shape)
    for i in range(1, user_history.shape[0]):
        cum_sum[i] = user_history[i - 1] + cum_sum[i - 1] * decay
    return normalize(cum_sum, norm, axis=1)


def task_profile(history: np.array, norm: str='l2'):
    return normalize(history.sum(axis=0).reshape(1, -1), norm).flatten()
