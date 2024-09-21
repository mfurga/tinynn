from __future__ import annotations
from typing import Tuple, Type, Union, Optional

import itertools
import numpy as np

Number = Union[int, float]


def _pad_left(*shapes: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
  max_dim = max(len(s) for s in shapes)
  return tuple((1,) * (max_dim - len(s)) + s for s in shapes)


class Tensor:
  def __init__(self,
               data,
               requires_grad: bool = False,
               creator: Optional[Function] = None):
    if isinstance(data, list):
      if all(isinstance(x, Tensor) for x in data):
        data = [x.data for x in data]

    if isinstance(data, Tensor):
      data = data.data

    self.data: np.array = np.array(data)
    self.requires_grad: bool = requires_grad

    self._grad: np.array = np.zeros_like(self.data, np.float64)
    self._creator: Optional[Function] = creator

  @property
  def grad(self) -> Tensor:
    return Tensor(self._grad)

  @property
  def shape(self) -> tuple:
    return self.data.shape

  @property
  def ndim(self) -> int:
    return self.data.ndim

  def zero_grad(self) -> None:
    self._grad = np.zeros_like(self.data, np.float64)

  def backward(self):
    self._grad = np.ones_like(self.data, np.float64)
    for t in reversed(self._topo_sort()):
      grads = t._creator.backward(t._grad)
      for child, grad in zip(t._creator.children, grads):
        child._grad += grad

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

  @classmethod
  def randn(cls, *shape: Tuple[int]) -> Tensor:
    return Tensor(np.random.randn(*shape))

  def assign(self, other: Tensor) -> None:
    self.data = other.data.copy()
    self.grad = other.grad.copy()
    self._creator = None

  def broadcast_to(self, shape: Tuple[int]) -> Tensor:
    return Broadcast.apply(self, shape=shape)

  def sum(self, axis: Optional[int] = None) -> Tensor:
    return Sum.apply(self, axis=axis)

  def mean(self, axis: Optional[int] = None) -> Tensor:
    return Mean.apply(self, axis=axis)

  def exp(self) -> Tensor:
    return Exp.apply(self)

  def dot(self, other: Tensor) -> Tensor:
    if self.ndim > 2 or other.ndim > 2:
      raise ValueError("Dot function supports only tensors up to 2D")
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

    if x.shape != y.shape:
      shape = tuple(0 if 0 in size else max(size)
        for size in zip(*_pad_left(x.shape, y.shape)))
      x = x.broadcast_to(shape)
      y = y.broadcast_to(shape)

    if reverse:
      x, y = y, x

    return x, y

  def equal(self, x: Tensor) -> bool:
    if not isinstance(x, Tensor):
      return False
    return np.array_equal(self.data, x.data)

  def __hash__(self) -> int:
    return id(self)

  def __getitem__(self, key) -> Tensor:
    return Tensor(self.data[key])

  def __neg__(self): return self.neg()

  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(x)
  def __mul__(self, x): return self.mul(x)
  def __truediv__(self, x): return self.div(x)
  def __matmul__(self, x): return self.dot(x)
  def __pow__(self, x): return self.pow(x)

  def __radd__(self, x): return self.add(x, True)
  def __rsub__(self, x): return self.sub(x, True)
  def __rmul__(self, x): return self.mul(x, True)
  def __rtruediv__(self, x): return self.div(x, True)
  def __rmatmul__(self, x): return self.dot(x, True)
  def __rpow__(self, x): return self.pow(x, True)

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
  def apply(cls: Type[Function], *x: Tensor, **kwargs) -> Tensor:
    func = cls(*x)
    y = func.forward(*[t.data for t in x], **kwargs)

    if func.requires_grad:
      ret = Tensor(y, requires_grad=True, creator=func)
    else:
      ret = Tensor(y, requires_grad=False)

    return ret


class Add(Function):
  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0 = x0
    self.x1 = x1
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
    if self.x0.ndim == 2 and self.x1.ndim == 1:
      gx0 = np.dot(np.atleast_2d(gy).T, np.atleast_2d(self.x1))
    else:
      gx0 = np.dot(gy, self.x1.T)

    if self.x0.ndim == 1 and self.x1.ndim == 2:
      gx1 = np.dot(np.atleast_2d(self.x0).T, np.atleast_2d(gy))
    else:
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


class Broadcast(Function):
  def forward(self, x0: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    self.org_shape = x0.shape
    self.shape = shape
    return np.broadcast_to(x0, shape)

  def backward(self, gy: np.array) -> np.ndarray:
    os, s = _pad_left(self.org_shape, self.shape)
    axis = tuple(i for i, (si, sj) in enumerate(zip(os, s)) if si != sj)
    gx0 = np.add.reduce(gy, axis=axis, keepdims=True).reshape(self.org_shape)
    return gx0,


class Sum(Function):
  def forward(self, x0: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    self.x0 = x0
    self.axis = axis
    return np.sum(x0, axis, keepdims=True)

  def backward(self, gy: np.array) -> np.ndarray:
    if self.axis is None:
      return np.full_like(self.x0, 1, np.float64) * gy,

    if self.axis == 0:
      g = np.full((n, 1), 1)
      return np.dot(g, gy),
    else:
      g = np.full((1, n), 1)
      return np.dot(gy, g),


class Mean(Function):
  def forward(self, x0: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    self.x0 = x0
    self.axis = axis
    return np.mean(x0, axis, keepdims=True)

  def backward(self, gy: np.array) -> np.ndarray:
    if self.axis is None:
      n = self.x0.size
      return np.full_like(self.x0, 1 / n, np.float64) * gy,

    n = self.x0.shape[self.axis]

    if self.axis == 0:
      g = np.full((n, 1), 1 / n)
      return np.dot(g, gy),
    else:
      g = np.full((1, n), 1 / n)
      return np.dot(gy, g),

