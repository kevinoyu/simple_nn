import numpy as np
from grad_val import GradVal


class GDOptimizer:
    def __init__(self, params: np.ndarray, lr=1e-3) -> None:
        self.params: np.ndarray = params
        self.lr = lr

    def optimize(self) -> None:
        for param in self.params:
            param.val -= param.gradient * self.lr
