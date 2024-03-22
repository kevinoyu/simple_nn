from simple_lib.grad_val import GradVal
import numpy as np
from model import Layer


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
