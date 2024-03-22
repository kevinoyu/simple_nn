from abc import ABC, abstractmethod
from typing import Iterable
from grad_val import GradVal
from random import random
import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, x: Iterable) -> Iterable:
        pass

    @abstractmethod
    def parameters(self) -> Iterable:
        pass

    def __call__(self, x: Iterable) -> Iterable:
        return self.forward(x)


class SimpleLinearLayer(Layer):
    def __init__(self, n_input: int, n_output: int) -> GradVal:
        self.shape = (n_input, n_output)

        self.weights: np.ndarray = np.array(
            [
                [GradVal(x) for x in np.random.uniform(-1, 1, n_input)]
                for _ in range(n_output)
            ]
            if n_output > 1
            else [GradVal(x) for x in np.random.uniform(-1, 1, n_input)]
        )

        self.bias: np.ndarray = np.random.uniform(-1, 1, n_output)

    def forward(self, x: np.ndarray[GradVal]) -> np.ndarray[GradVal]:
        return x @ self.weights.T + self.bias

    def parameters(self) -> np.ndarray[GradVal]:
        return np.ndarray([self.weights.flatten(), self.bias.flatten()])


class SimpleReluLayer(Layer):
    def __init__(self):
        pass

    def forward(self, x: np.ndarray[GradVal]) -> np.ndarray[GradVal]:
        return np.array([[val.relu() for val in _x] for _x in x])

    def parameters(self) -> list[GradVal]:
        return []


class Model(Layer):
    def __init__(self, pipeline: Iterable[Layer]):
        self.pipeline: Iterable[Layer] = pipeline

    def forward(self, x: np.ndarray[GradVal]) -> np.ndarray[GradVal]:
        for layer in self.pipeline:
            x = layer.forward(x)
        return x

    def parameters(self) -> np.ndarray[GradVal]:
        params = np.array([])
        for layer in self.pipeline:
            params = np.append(params, layer.parameters())
        return params
