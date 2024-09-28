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

  @classmethod
  def randn(cls, *shape: Tuple[int]) -> Tensor:
    return Tensor(np.random.randn(*shape))

  @classmethod
  def zeros(cls, shape: Tuple[int, ...]) -> Tensor:
    return Tensor(np.zeros(shape))

  @property
  def grad(self) -> Tensor:
    return Tensor(self._grad)

  @property
  def shape(self) -> tuple:
    return self.data.shape

  @property
  def ndim(self) -> int:
    return self.data.ndim

  @property
  def size(self) -> int:
    return self.data.size

  def zero_grad(self) -> None:
    self._grad = np.zeros_like(self.data, np.float64)

  def backward(self, grad: Optional[Tensor] = None):
    self._grad = np.ones_like(self.data, np.float64)
    if grad is not None:
      if grad.shape != self.shape:
        raise Exception("Invalid grad shape")
      self._grad = grad.data

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

  def assign(self, other: Tensor) -> None:
    self.data = other.data.copy()
    self._grad = other._grad.copy()
    self._creator = None

  def broadcast_to(self, shape: Tuple[int]) -> Tensor:
    return Broadcast.apply(self, shape=shape)

  def max(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    return Max.apply(self, axis=axis, keepdims=keepdims)

  def min(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    return Min.apply(self, axis=axis, keepdims=keepdims)

  def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    return Sum.apply(self, axis=axis, keepdims=keepdims)

  def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    n = self.size
    if axis is not None:
      n = self.shape[axis]
    return self.sum(axis=axis, keepdims=keepdims) / n

  def log(self) -> Tensor:
    return Log.apply(self)

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
    raise NotImplementedError("forward method is not implemented")

  def backward(self, *args):
    raise NotImplementedError("backward method is not implemented")

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
    self.res = np.reciprocal(x0, dtype=np.float64)
    return self.res

  def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
    gx0 = -gy * self.res * self.res
    return gx0,


class Log(Function):
  def forward(self, x0: np.ndarray) -> np.ndarray:
    self.x0 = x0
    return np.log(x0)

  def backward(self, gy: np.array) -> np.ndarray:
    gx0 = gy / self.x0
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


class Max(Function):
  def forward(self, x: np.ndarray, axis: Optional[int] = None,
              keepdims: bool = False) -> np.ndarray:
    self.x = x
    self.axis = axis
    self.keepdims = keepdims
    return np.max(x, axis=axis, keepdims=keepdims)

  def backward(self, gy: np.array) -> np.ndarray:
    max = self.x.max(axis=self.axis, keepdims=True)
    max_1s = (self.x == max).astype(int)
    if self.axis and not self.keepdims:
      gy = np.expand_dims(gy, axis=self.axis)
    return (max_1s / np.sum(max_1s, self.axis, keepdims=True)) * gy,


class Min(Function):
  def forward(self, x: np.ndarray, axis: Optional[int] = None,
              keepdims: bool = False) -> np.ndarray:
    self.x = x
    self.axis = axis
    self.keepdims = keepdims
    return np.min(x, axis=axis, keepdims=keepdims)

  def backward(self, gy: np.array) -> np.ndarray:
    min = self.x.min(axis=self.axis, keepdims=True)
    min_1s = (self.x == min).astype(int)
    if self.axis and not self.keepdims:
      gy = np.expand_dims(gy, axis=self.axis)
    return (min_1s / np.sum(max_1s, self.axis, keepdims=True)) * gy,


class Sum(Function):
  def forward(self, x: np.ndarray, axis: Optional[int] = None,
              keepdims: bool = False) -> np.ndarray:
    self.x = x
    self.axis = axis
    self.keepdims = keepdims
    return np.sum(x, axis=axis, keepdims=keepdims)

  def backward(self, gy: np.array) -> np.ndarray:
    if self.axis and not self.keepdims:
      gy = np.expand_dims(gy, axis=self.axis)
    return np.broadcast_to(gy, self.x.shape),

