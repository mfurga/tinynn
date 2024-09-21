#!/usr/bin/env python3

from typing import Callable, List

from tinynn import Tensor

def sigmoid(x: Tensor) -> Tensor:
  return 1. / (1. + (-x).exp())


class MSELoss:
  def __call__(self, x: Tensor, y: Tensor) -> Tensor:
    return ((x - y) ** 2).mean()


class SGD:
  def __init__(self, params: List[Tensor], lr: float = 0.001):
    for param in params:
      param.requires_grad = True
    self.params = params
    self.lr = lr

  def zero_grad(self) -> None:
    for param in self.params:
      param.zero_grad()

  def step(self) -> None:
    for param in self.params:
      g = param.grad
      param.assign(param - self.lr * g)

