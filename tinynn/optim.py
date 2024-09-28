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
  def __init__(self, params: List[Tensor], lr: float = 0.001,
               momentum: float = 0):
    super().__init__(params)
    self.lr = lr
    self.momentum = momentum
    self.m = [Tensor.zeros(p.shape) for p in params] if momentum else []

  def step(self) -> None:
    for i, param in enumerate(self.params):
      g = param.grad
      g = self.lr * g

      if self.momentum:
        self.m[i] = self.momentum * self.m[i] + g
        g = self.m[i]

      param.assign(param - g)

