from __future__ import annotations
from grad_vector import GradVector
import numpy as np
from abc import ABC, abstractmethod


class VectorLayer(ABC):
    @abstractmethod
    def forward(self, x: GradVector) -> GradVector:
        pass

    @abstractmethod
    def parameters(self) -> np.ndarray[GradVector]:
        pass

    def __call__(self, x: GradVector) -> GradVector:
        return self.forward(x)


class VectorLinearLayer(VectorLayer):
    def __init__(self, n_input: int, n_output: int) -> None:
        self.shape = (n_input, n_output)
        self.weights: GradVector = GradVector(np.random.rand(n_input, n_output))
        self.bias: GradVector = GradVector(np.random.rand(n_output))

    def forward(self, x: GradVector) -> GradVector:
        return x @ self.weights + self.bias

    def parameters(self) -> np.ndarray[GradVector]:
        return np.array([self.weights, self.bias])

class VectorReluLayer(VectorLayer):
    def forward(self, x: GradVector) -> GradVector:
        return x.relu()
    
    def parameters(self) -> np.ndarray[GradVector]:
        return np.array([])


