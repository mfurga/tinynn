from typing import List

from tinynn import Tensor

class _Loss:
  def __call__(self, input: Tensor, target: Tensor) -> Tensor:
    raise NotImplementedError("__call__ method is not implemented")


class MSELoss(_Loss):
  def __call__(self, input: Tensor, target: Tensor) -> Tensor:
    return ((input - target) ** 2).mean()


class Module:
  def __call__(self, x: Tensor) -> Tensor:
    return self.forward(x)

  def parameters(self) -> List[Tensor]:
    params = []
    for attr in self.__dict__.values():
      if isinstance(attr, Module):
        params.extend(attr.parameters())
      if isinstance(attr, Tensor):
        params.append(attr)
    return params

  def forward(self, x: Tensor) -> Tensor:
    raise NotImplementedError("forward method is not implemented")


class Linear(Module):
  def __init__(self, in_size: int, out_size: int, bias: bool = True) -> None:
    self.weights = Tensor.randn(in_size, out_size)
    self.bias = bias
    if bias:
      self.biases = Tensor.randn(out_size)

  def forward(self, x: Tensor) -> Tensor:
    r = x.dot(self.weights)
    if self.bias:
      r += self.biases
    return r

