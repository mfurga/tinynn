from typing import List

from tinynn import Tensor

class Optimizer:
  def __init__(self, params: List[Tensor]):
    for param in params:
      param.requires_grad = True
    self.params = params

  def zero_grad(self) -> None:
    for param in self.params:
      param.zero_grad()

  def step(self) -> None:
    raise NotImplementedError("step method is not implemented")


class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr: float = 0.001):
    super().__init__(params)
    self.lr = lr

  def step(self) -> None:
    for param in self.params:
      g = param.grad
      param.assign(param - self.lr * g)

