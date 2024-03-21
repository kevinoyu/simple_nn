from abc import ABC, abstractmethod
from typing import Iterable
from grad_val import GradVal
from random import random
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray[float]) -> np.ndarray[GradVal]:
        pass

    @abstractmethod
    def parameters(self) -> np.ndarray[GradVal]:
        pass


class BiasedLinearLayer:
    def __init__(self, n_input: int, n_output: int) -> GradVal:
        self.shape = (n_input, n_output)

        self.weights: np.ndarray = np.array(
            [[GradVal(random()) for _ in range(n_input)] for _ in range(n_output)]
            if n_output > 1
            else [GradVal(random()) for _ in range(n_input)]
        )

        self.bias: np.ndarray = np.array([GradVal(random()) for _ in range(n_output)])

    def forward(self, x: np.ndarray[float]) -> np.ndarray[GradVal]:
        return x @ self.weights.T + self.bias

    def parameters(self) -> np.ndarray[GradVal]:
        return self.weights.flatten()


class ReluLayer:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray[float]) -> np.ndarray[GradVal]:
        return np.array([[val.relu() for val in _x] for _x in x])

    def parameters(self) -> list[GradVal]:
        return []


class Model:
    def __init__(self, pipeline: Iterable[Layer]):
        self.pipeline: Iterable[Layer] = pipeline

    def forward(self, x: np.ndarray[float]) -> np.ndarray[GradVal]:
        for layer in self.pipeline:
            x = layer.forward(x)
        return x

    def parameters(self) -> np.ndarray[GradVal]:
        params = np.array([])
        for layer in self.pipeline:
            params = np.append(params, layer.parameters())
        return params
