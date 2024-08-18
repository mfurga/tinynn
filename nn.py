#!/usr/bin/env python3

from typing import Callable

import random
import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def dsigmoid(z):
  return sigmoid(z) * (1. - sigmoid(z))

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
  def cost_derivative(y: np.array, v: np.array) -> np.array:
    """
    Cost function for single training input.
      y - result from the network (vector)
      v - actual value (vector)

      C(y, v) = 0.5 || y - v || ^ 2 = 0.5 * sum[ yi - vi ] ^ 2
      grad C = y - v
    """
    return y - v

  def train(self,
            training_data: np.array,
            learning_rate: float = 0.01,
            epochs: int = 100,
            mini_batch_size: int = 4):
    n = len(training_data)
    m = max(epochs // 10, 1)

    for epoch in range(1, epochs + 1):
      random.shuffle(training_data)

      mini_batches = [
        training_data[i: i + mini_batch_size]
        for i in range(0, n, mini_batch_size)
      ]

      for mini_batch in mini_batches:
        self._train_mini_batch(mini_batch, learning_rate)

      if epoch % m == 0:
        loss = self.total_cost(training_data)
        print(f"[{epoch} / {epochs} epoch] loss = {loss:.5}")

  def _train_mini_batch(self,
                        mini_batch: list,
                        learning_rate: float) -> None:
    m = len(mini_batch)
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    for (x, v) in mini_batch:
      delta_nabla_w, delta_nabla_b = self.backprop(x, v)
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

    self.weights = [w - learning_rate * (nw / m)
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - learning_rate * (nb / m)
                   for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x: np.array, v: np.array):
    preactivations = [0]
    activations = [x]
    a = x

    for w, b in zip(self.weights, self.biases):
      z = w.dot(a) + b
      preactivations.append(z)
      a = sigmoid(z)
      activations.append(a)

    nabla_a = [np.zeros(a.shape) for a in activations]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    nabla_a[-1] = NN.cost_derivative(activations[-1], v)

    for l in range(self.num_layers - 2, -1, -1):
      t = nabla_a[l + 1] * dsigmoid(preactivations[l + 1])

      nabla_w[l] = np.dot(t, activations[l].T)
      nabla_b[l] = t
      nabla_a[l] = np.dot(self.weights[l].T, t)

    return (nabla_w, nabla_b)

  def total_cost(self, data: np.array) -> float:
    n = len(data)
    cost = 0.
    for (x, v) in data:
      y = self.feedforward(x)
      cost += 0.5 * sum((y - v) ** 2)[0]
    return cost / n

def main():
  np.random.seed(0)
  nn = NN((3, 2, 1))

  x = np.array([0., 0.2, 0.4]).reshape(-1, 1)
  v = np.array([0.3]).reshape(-1, 1)

  training_data = [(x, v)]

  nn.train(
    training_data,
    learning_rate=0.1,
    epochs=1000
  )

  print(nn.feedforward(x))

if __name__ == "__main__":
  main()

