import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize


def rowwise_cosine(x, y):
  """
  https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
  """
  return - np.einsum('ij,ij->i', x, y) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

  
def rowwise_euclidean(x, y):
  return np.sqrt(np.sum(np.square(x - y), axis=1))


class DistanceModel:
    def __init__(self, metric: str='euclidean', msg_norm: Optional[str]='l2', model: BaseEstimator=None):
        """
        metric - one of euclidean / cosine
        msg_norm - one of None / l2 / l1 / max
        """
        self.metric = metric
        self.msg_norm = msg_norm
        self.model = model


    def fit(self, profiles: np.array, messages: np.array):
        if self.model is not None:
            self.model_ = self.model.fit(profiles, messages)
        return self
    

    def predict(self, profiles: np.array, messages: np.array):
        if self.model is not None:
            profiles = self.model_.predict(profiles)
        if self.msg_norm:
              messages = normalize(messages, self.msg_norm, axis=1)
        if self.metric == 'euclidean':
            return rowwise_euclidean(profiles, messages)
        elif self.metric == 'cosine':
            return rowwise_cosine(profiles, messages)
        raise RuntimeError('Metric not recognized!')
    

    def fit_predict(self, profiles, messages):
        return self.fit(profiles, messages).predict(profiles, messages)