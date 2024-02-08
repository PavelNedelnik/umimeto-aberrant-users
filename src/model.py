import numpy as np
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize


def rowwise_cosine(x, y):
  """
  https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
  TODO smoothing constant!
  """
  return 1 - np.einsum('ij,ij->i', x, y) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

  
def rowwise_euclidean(x, y):
  return np.sqrt(np.sum(np.square(x - y), axis=1))


class DistanceModel:
    def __init__(self, metric: str='euclidean', msg_norm: Optional[str]='l2'):
        self.metric = metric
        self.msg_norm = msg_norm
    

    def calculate_distances(self, profiles: pd.Series, messages: pd.Series) -> pd.Series:
        index = profiles.index  # TODO might not match messages!
        profiles = np.vstack(profiles)
        messages =  np.vstack(messages)
        if self.msg_norm:
              messages = normalize(messages, self.msg_norm, axis=1)
        if self.metric == 'euclidean':
            distances = rowwise_euclidean(profiles, messages)
        elif self.metric == 'cosine':
            distances = rowwise_cosine(profiles, messages)
        else:
            raise RuntimeError('Metric not recognized!')
        
        return pd.Series(distances, index=index)
    