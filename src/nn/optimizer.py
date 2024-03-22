from collections.abc import Iterable


class GDOptimizer:
    def __init__(self, params: Iterable, lr=1e-3) -> None:
        self.params: Iterable = params
        self.lr = lr

    def optimize(self) -> None:
        for param in self.params:
            param.val -= param.gradient * self.lr
