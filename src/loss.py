import numpy as np
from grad_val import GradVal
from typing import Iterable


class MSELoss:

    def __init__(self):
        pass

    @staticmethod
    def loss(y_pred: np.ndarray[GradVal], y_true: np.ndarray[GradVal]) -> GradVal:
        diff = y_pred - y_true
        if isinstance(diff.T, np.ndarray):
            return diff.T @ diff / np.prod(diff.shape)
        else:
            return diff.T() @ diff / np.prod(diff.shape)
