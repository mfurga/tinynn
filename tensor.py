from __future__ import annotations
from typing import Tuple, Type, Union, Optional

import numpy as np

Number = Union[int, float]

class Tensor:
  def __init__(self,
               data: np.ndarray,
               requires_grad: bool = False,
               creator: Optional[Function] = None):
    self.data: np.array = np.array(data)
    self.requires_grad: bool = requires_grad
    self.grad: Optional[float] = np.zeros_like(data, np.float64)

    self._creator: Optional[Function] = creator

  @property
  def shape(self) -> tuple:
    return self.data.shape

  def backward(self):
    self.grad = np.ones_like(self.data, np.float64)
    for t in reversed(self._topo_sort()):
      grads = t._creator.backward(t.grad)
      for child, grad in zip(t._creator.children, grads):
        child.grad += grad

  def _topo_sort(self) -> List[Tensor]:
    visited: Set[Tensor] = set()
    topo: List[Tensor] = []

    def build_topo(v: Tensor):
      if v in visited:
        return
      visited.add(v)
      if v._creator:
        for child in v._creator.children:
          build_topo(child)
        topo.append(v)

    build_topo(self)
    return topo

  def exp(self) -> Tensor:
    return Exp.apply(self)

  def dot(self, other: Tensor) -> Tensor:
    return Dot.apply(self, other)

  def reciprocal(self) -> Tensor:
    return Reciprocal.apply(self)

  def neg(self) -> Tensor:
    return self * (-1)

  def add(self, other: Union[Tensor, Number], reverse: bool = False) -> Tensor:
    return Add.apply(*self._normalize(other, reverse))

  def sub(self, other: Union[Tensor, Number], reverse: bool = False) -> Tensor:
    x, y = self._normalize(other, reverse)
    return x + (-y)

  def mul(self, other: Union[Tensor, Number], reverse: bool = False) -> Tensor:
    return Mul.apply(*self._normalize(other, reverse))

  def div(self, other: Union[Tensor, Number], reverse: bool = False) -> Tensor:
    x, y = self._normalize(other, reverse)
    return x * y.reciprocal()

  def pow(self, other: Union[Tensor, Number], reverse: bool = False) -> Tensor:
    return Pow.apply(*self._normalize(other, reverse))

  def _normalize(self, y: Union[Tensor, Number],
                 reverse: bool = False) -> Tensor:
    x = self

    if isinstance(y, Number):
      y = Tensor(np.full(self.shape, y))

    if reverse:
      x, y = y, x

    return x, y

  def __neg__(self): return self.neg()

  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(other)
  def __mul__(self, x): return self.mul(other)
  def __truediv__(self, x): return self.div(other)
  def __matmul__(self, x): return self.dot(other)
  def __pow__(self, x): return self.pow(other)

  def __radd__(self, x): return self.add(other, True)
  def __rsub__(self, x): return self.sub(other, True)
  def __rmul__(self, x): return self.mul(other, True)
  def __rtruediv__(self, x): return self.div(other, True)
  def __rmatmul__(self, x): return self.dot(other, True)
  def __rpow__(self, x): return self.pow(other, True)

  def __repr__(self):
    return f"{self.data}"


class Function:
  def __init__(self, *x: Tensor):
    self.requires_grad = any(t.requires_grad for t in x)
    self.children: Tuple[Tensor] = (),

    if self.requires_grad:
      self.children = x

  def forward(self, *args):
    raise NotImplementedError("Forward function is not implemented")

  def backward(self, *args):
    raise NotImplementedError("Backward function is not implemented")

  @classmethod
  def apply(cls: Type[Function], *x: Tensor) -> Tensor:
    func = cls(*x)
    y = func.forward(*[t.data for t in x])

    if func.requires_grad:
      ret = Tensor(y, requires_grad=True, creator=func)
    else:
      ret = Tensor(y, requires_grad=False)

    return ret


class Add(Function):
  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return x0 + x1

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = gy
    gx1 = gy
    return gx0, gx1


class Mul(Function):
  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0 = x0
    self.x1 = x1
    return x0 * x1

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = self.x1 * gy
    gx1 = self.x0 * gy
    return gx0, gx1


class Pow(Function):
  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0 = x0
    self.x1 = x1
    return x0 ** x1

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = (self.x1 * (self.x0 ** (self.x1 - 1))) * gy
    gx1 = (self.x0 ** self.x1 * np.log(self.x0)) * gy
    return gx0, gx1


class Dot(Function):
  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0 = x0
    self.x1 = x1
    return np.dot(x0, x1)

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = np.dot(gy, self.x1.T)
    gx1 = np.dot(self.x0.T, gy)
    return gx0, gx1


class Reciprocal(Function):
  def forward(self, x0: np.ndarray) -> np.ndarray:
    self.x0 = x0
    return 1. / x0

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = - (1 / self.x0 ** 2)
    return gx0,


class Exp(Function):
  def forward(self, x0: np.ndarray) -> np.ndarray:
    self.x0 = x0
    return np.exp(x0)

  def backward(self, gy: np.array) -> np.ndarray:
    gx0 = np.exp(self.x0) * gy
    return gx0,

