from __future__ import annotations
from typing import Callable, Iterable, Union
import graphviz
import numpy as np
from iota import iota
from random import random


class GradVector:
    def __init__(
        self, vals: Iterable, ancestors: tuple = (), op: str = "", label: str = None
    ) -> None:
        self.val: np.ndarray[float] = np.array(vals)
        self.shape = self.val.shape
        self.gradient: np.ndarray[float] = np.zeros(self.val.shape)
        self._back: Callable = lambda: None
        self.ancestors = ancestors
        self._operation = op
        self.label = label if label else iota()

    def __add__(self, other: Union[GradVector, float, int]) -> GradVector:
        other = (
            other
            if isinstance(other, GradVector)
            else GradVector(np.zeros(self.shape) + other)
        )

        new_tensor = GradVector(self.val + other.val, (self, other), op="+")

        if other.shape == self.shape:

            def _back_closure():
                other.gradient += new_tensor.gradient
                self.gradient += new_tensor.gradient

            new_tensor._back = _back_closure
        elif len(self.shape) != len(other.shape) and self.shape[1:] == other.shape:

            def _back_closure():
                self.gradient += new_tensor.gradient
                other.gradient += np.sum(new_tensor.gradient, axis=0)

            new_tensor._back = _back_closure
        else:
            return None

        return new_tensor

    def __sub__(self, other: Union[GradVector, float, int]) -> GradVector:
        return self + -other

    def __rsub__(self, other: Union[float, int]) -> GradVector:
        return self - other

    def __neg__(self) -> GradVector:
        return self * -1.0

    def __radd__(self, other: Union[float, int]) -> GradVector:
        return self + other

    def __mul__(self, other: Union[float, int]) -> GradVector:
        assert isinstance(other, (int, float))
        new_tensor = GradVector(self.val * other, (self,), op=f"*{other}")

        def _back_closure():
            self.gradient += other

        new_tensor._back = _back_closure

        return new_tensor

    def __rmul__(self, other) -> GradVector:
        return self * other

    def __truediv__(self, other: Union[float, int]) -> GradVector:
        return self * other**-1.0

    def __rtruediv__(self, other: Union[float, int]) -> GradVector:
        return self / other

    def __pow__(self, other: Union[float, int]) -> GradVector:
        new_tensor = GradVector(self.val**other, (self,), op=f"^{other}")

        def _back_closure():
            self.gradient += np.multiply(
                new_tensor.gradient * other, self.val ** (other - 1)
            )

        new_tensor._back = _back_closure

        return new_tensor

    def __matmul__(self, other: GradVector) -> GradVector:
        new_tensor = GradVector(self.val @ other.val, (self, other), op="@")

        def _back_closure():
            self.gradient += new_tensor.gradient @ other.val.T
            other.gradient += self.val.T @ new_tensor.gradient

        new_tensor._back = _back_closure

        return new_tensor

    def relu(self) -> GradVector:
        new_tensor = GradVector(np.clip(self.val, 0, None), (self,), op="relu")

        def _back_closure():
            self.gradient += np.where(self.val > 0, new_tensor.gradient, 0)

        new_tensor._back = _back_closure
        return new_tensor

    def T(self) -> GradVector:
        new_tensor = GradVector(vals=self.val.T, ancestors=(self,), op="T")

        def _back_closure():
            self.gradient += new_tensor.gradient.T

        new_tensor._back = _back_closure

        return new_tensor

    def backward(self) -> None:
        order: list[GradVector] = []
        visited: set[GradVector] = set()

        def search(vertex: GradVector):
            if vertex in visited:
                return
            visited.add(vertex)
            for ancestor in vertex.ancestors:
                search(ancestor)
            order.append(vertex)

        search(self)

        self.gradient = np.ones(self.shape)
        for vertex in reversed(order):
            vertex._back()

    def zero_grad(self) -> None:
        visited: set[GradVector] = set()

        def search(vertex: GradVector):
            if vertex in visited:
                return
            vertex.gradient = np.zeros(vertex.shape)
            for ancestor in vertex.ancestors:
                search(ancestor)

    def __repr__(self) -> str:
        return f"{self.val}"

    def visualize(self) -> graphviz.Digraph:
        dot = graphviz.Digraph()
        vertices: set[GradVector] = set()
        edges: set[tuple[GradVector]] = set()

        def search(vertex: GradVector) -> None:
            if vertex in vertices:
                return
            vertices.add(vertex)
            for ancestor in vertex.ancestors:
                edges.add((ancestor, vertex))
                search(ancestor)

        def vid(vertex: GradVector) -> str:
            return str(id(vertex))

        def oid(vertex: GradVector) -> str:
            return vid(vertex=vertex) + vertex._operation

        search(self)

        for vertex in vertices:
            vertex_id: str = vid(vertex=vertex)

            dot.node(
                name=vertex_id,
                label=f"{vertex.label} | {vertex.val} grad={vertex.gradient}",
            )

            if vertex._operation:
                op_id: str = oid(vertex=vertex)
                dot.node(name=op_id, label=vertex._operation)
                dot.edge(op_id, vertex_id)

        for out_vertex, in_vertex in edges:
            dot.edge(vid(vertex=out_vertex), oid(vertex=in_vertex))

        return dot
