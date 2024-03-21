import numpy as np
from grad_val import GradVal
from typing import Iterable


class MSELoss:

    def __init__(self):
        pass

    @staticmethod
    def loss(y_pred: np.ndarray[GradVal], y_true: np.ndarray[GradVal]) -> GradVal:
        assert len(y_pred) == len(y_true)
        return np.sum((y_pred - y_true) ** 2) / len(y_pred)
