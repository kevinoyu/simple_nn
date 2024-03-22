from __future__ import annotations
from gradient.grad_vector import GradVector
import numpy as np
from nn.model import Layer

class VectorLinearLayer(Layer):
    def __init__(self, n_input: int, n_output: int) -> None:
        self.shape = (n_input, n_output)
        self.weights: GradVector = GradVector(
            np.random.uniform(-1, 1, (n_input, n_output))
        )
        self.bias: GradVector = GradVector(np.random.uniform(-1, 1, n_output))

    def forward(self, x: GradVector) -> GradVector:
        return x @ self.weights + self.bias

    def parameters(self) -> np.ndarray[GradVector]:
        return np.array([self.weights, self.bias])


class VectorReluLayer(Layer):
    def forward(self, x: GradVector) -> GradVector:
        return x.relu()

    def parameters(self) -> np.ndarray[GradVector]:
        return np.array([])
