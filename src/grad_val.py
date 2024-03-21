from __future__ import annotations
import math
from typing import Union, Callable
import graphviz
from iota import iota

class GradVal:
    def __init__(self, val: float, ancestors: tuple = (), op: str = "") -> None:
        self.val: float = val
        self.gradient: float = 0
        self.ancestors: tuple = ancestors
        self._operation: str = op
        self._back: Callable = lambda: None
        self.label = iota()

    def __repr__(self) -> str:
        return f"{self.val}"

    def __add__(self, other: Union[int, float, GradVal]) -> GradVal:
        other = other if isinstance(other, GradVal) else GradVal(other)

        new_param: GradVal = GradVal(
            val=self.val + other.val,
            ancestors=(self, other),
            op="+",
        )

        def _back_closure():
            other.gradient += new_param.gradient
            self.gradient += new_param.gradient

        new_param._back = _back_closure

        return new_param

    def __radd__(self, other: float) -> GradVal:
        return self + other

    def __mul__(self, other: Union[float, GradVal]) -> GradVal:
        other = other if isinstance(other, GradVal) else GradVal(other)

        new_param: GradVal = GradVal(
            val=other.val * self.val,
            ancestors=(self, other),
            op="*",
        )

        def _back_closure():
            other.gradient += new_param.gradient * self.val
            self.gradient += new_param.gradient * other.val

        new_param._back = _back_closure

        return new_param

    def __rmul__(self, other: float) -> GradVal:
        return self * other

    def __truediv__(self, other: Union[int, float, GradVal]) -> GradVal:
        return self * other**-1.0

    def __neg__(self) -> GradVal:
        return self * -1

    def __sub__(self, other) -> GradVal:
        return self + -other

    def exp(self) -> GradVal:
        new_param: GradVal = GradVal(
            val=math.exp(self.val),
            ancestors=(self,),
            op="exp",
        )

        def _back_closure():
            self.gradient += new_param.gradient * new_param.val

        new_param._back = _back_closure

        return new_param

    def __pow__(self, other: Union[int, float]) -> None:
        new_param: GradVal = GradVal(
            val=self.val**other,
            ancestors=(self,),
            op=f"pow_{other}",
        )

        def _back_closure():
            self.gradient += new_param.gradient * other * self.val ** (other - 1)

        new_param._back = _back_closure

        return new_param

    def relu(self) -> GradVal:
        new_param: GradVal = GradVal(val=max(0, self.val), ancestors=(self,), op="relu")

        def _back_closure():
            self.gradient = 0 if self.val <= 0 else new_param.gradient

        new_param._back = _back_closure

        return new_param

    def backward(self) -> None:
        order: list[GradVal] = []
        visited: set[GradVal] = set()

        def search(vertex: GradVal):
            if vertex in visited:
                return
            visited.add(vertex)
            for ancestor in vertex.ancestors:
                search(ancestor)
            order.append(vertex)

        search(self)

        self.gradient = 1.0
        for vertex in reversed(order):
            vertex._back()

    def zero_grad(self) -> None:
        visited: set[GradVal] = set()

        def _zero(vertex: GradVal) -> None:
            if vertex in visited:
                return
            vertex.gradient = 0.0
            for ancestor in vertex.ancestors:
                _zero(ancestor)

        _zero(self)

    def visualize(self) -> graphviz.Digraph:
        dot = graphviz.Digraph()
        vertices: set[GradVal] = set()
        edges: set[tuple[GradVal]] = set()

        def search(vertex: GradVal) -> None:
            if vertex in vertices:
                return
            vertices.add(vertex)
            for ancestor in vertex.ancestors:
                edges.add((ancestor, vertex))
                search(ancestor)

        def vid(vertex: GradVal) -> str:
            return str(id(vertex))

        def oid(vertex: GradVal) -> str:
            return vid(vertex=vertex) + vertex._operation

        search(self)

        for vertex in vertices:
            vertex_id: str = vid(vertex=vertex)

            dot.node(
                name=vertex_id,
                label=f"{vertex.label} | {vertex.val:.3f} grad={vertex.gradient:.3f}",
            )

            if vertex._operation:
                op_id: str = oid(vertex=vertex)
                dot.node(name=op_id, label=vertex._operation)
                dot.edge(op_id, vertex_id)

        for out_vertex, in_vertex in edges:
            dot.edge(vid(vertex=out_vertex), oid(vertex=in_vertex))

        return dot
