from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np

class Layer(ABC):
    @abstractmethod
    def forward(self, x: any) -> any:
        pass

    @abstractmethod
    def parameters(self) -> any:
        pass

    def __call__(self, x: any) -> any:
        return self.forward(x)

class Pipeline(Layer):
    def __init__(self, pipeline: Iterable[Layer]):
        self.pipeline: Iterable[Layer] = pipeline

    def forward(self, x: Iterable) -> Iterable:
        for layer in self.pipeline:
            x = layer.forward(x)
        return x

    def parameters(self) -> Iterable:
        params = np.array([])
        for layer in self.pipeline:
            params = np.append(params, layer.parameters())
        return params
