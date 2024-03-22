import numpy as np
from typing import Iterable


class MSELoss:

    def __init__(self):
        pass

    @staticmethod
    def loss(y_pred: Iterable, y_true: Iterable) -> Iterable:
        diff = y_pred - y_true
        try:  # ugly.....
            return (diff.T @ diff) / np.prod(diff.shape)
        except:
            return (diff.T() @ diff) / np.prod(diff.shape)
