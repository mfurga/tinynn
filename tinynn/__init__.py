from typing import Optional

from tinynn.tensor import Tensor


def sigmoid(x: Tensor) -> Tensor:
  return 1. / (1. + (-x).exp())


def softmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
  if axis is None:
    axis = x.ndim - 1
  ezp = (x - x.max(axis=axis, keepdims=True)).exp()
  return ezp / ezp.sum(axis=axis, keepdims=True)

