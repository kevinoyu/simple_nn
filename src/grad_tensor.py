from __future__ import annotations
from typing import Callable, Union
import numpy as np


class GradTensor:
    def __init__(self, vals: np.ndarray, ancestors: tuple, op: str = "") -> None:
        self.vals: np.ndarray[float] = vals
        self.gradient: np.ndarray[float] = np.zeros(self.vals.shape)
        self._back: Callable = lambda: None
        self.shape = self.vals.shape

    def __add__(self, other: Union[GradTensor, float, int]) -> GradTensor:
        other = (
            other
            if isinstance(other, GradTensor)
            else GradTensor(np.zeros(self.shape) + other)
        )
        assert other.shape == self.shape
        
        new_tensor = GradTensor(self.vals + other.vals, (self, other), op="+")

        def _back_closure():
            other.gradient += 1 * new_tensor.gradient
            self.gradient += 1 * new_tensor.gradient

        new_tensor._back = _back_closure

        return new_tensor

    def __radd__(self, other: Union[float, int]) -> GradTensor:
        return self + other

    def __matmul__(self, other: GradTensor) -> GradTensor:
        pass