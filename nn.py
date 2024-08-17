#!/usr/bin/env python3

from typing import Callable

import numpy as np

def sigmoid(z):
  return 1. + (1. + np.exp(-z))

def dsigmoid(z):
  return sigmoid(z) * (1 - sigmoid(z))

class NN:
  def __init__(self, size: tuple):
    self.size = size
    self.num_layers = len(size)
    self.weights = [
      np.random.randn(y, x) for (x, y) in zip(size[:-1], size[1:])
    ]
    self.biases = [np.random.randn(x, 1) for x in size[1:]]

  def feedforward(self, a: np.array) -> np.array:
    for w, b in zip(self.weights, self.biases):
      a = sigmoid(w.dot(a) + b)
    return a

  @staticmethod
  def cost_derivative(a: np.array, y: np.array) -> np.array:
    """
      a - result from the network (vector)
      y - actual value (vector)

      C(a, y) = 0.5 || a - y || ^ 2 = 0.5 * sum[ ai - yi ] ^ 2
      grad C = a - y
    """
    return a - y

  def backprop(self, a: np.array, y: np.array):
    preactivations = [0]
    activations = [a]

    for w, b in zip(self.weights, self.biases):
      z = w.dot(a) + b
      preactivations.append(z)
      a = sigmoid(z)
      activations.append(a)

    nabla_a = [np.zeros(a.shape) for a in activations]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    nabla_a[-1] = NN.cost_derivative(activations[-1], y)

    for l in range(self.num_layers - 2, -1, -1):
      t = nabla_a[l + 1] * dsigmoid(preactivations[l + 1])

      nabla_w[l] = np.dot(t, activations[l].T)
      nabla_b[l] = t
      nabla_a[l] = np.dot(self.weights[l].T, t)

    return (nabla_w, nabla_b)

def main():
  np.random.seed(0)
  nn = NN((3, 4, 2, 3))

  a = np.array([1, 1, 1]).reshape(-1, 1)
  y = np.array([2, 2, 2]).reshape(-1, 1)
  grad = nn.backprop(a, y)
  print(grad)

if __name__ == "__main__":
  main()

