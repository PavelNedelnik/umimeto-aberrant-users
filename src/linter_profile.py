import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from typing import Optional

def freq_profile(linter_messages: np.array, task_means: Optional[np.array]=None, norm: str='l2', decay: float=1.):
    """
    history is sorted by timestamp
    norm: one of l1, l2, max
    decay: 1 means no decay, 0 means maximum decay
    """
    if task_means is None:
        task_means = np.zeros(linter_messages.shape)
    cum_sum = np.zeros(linter_messages.shape)
    for i in range(1, linter_messages.shape[0]):
        cum_sum[i] = cum_sum[i - 1] * decay + linter_messages[i - 1] - task_means[i - 1]
    return normalize(cum_sum, norm, axis=1)


def task_profile(history: np.array, norm: str='l2'):
    return normalize(history.sum(axis=0).reshape(-1, 1), norm, axis=0).flatten()
