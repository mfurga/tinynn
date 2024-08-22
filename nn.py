#!/usr/bin/env python3

from typing import Callable
from mnist_loader import mnist_load, raw_load

import random
import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def dsigmoid(z):
  return sigmoid(z) * (1. - sigmoid(z))

class Network:
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
    Derivative of cost function C(y, v).
    C(y, v) = 0.5 || y - v || ^ 2 = 0.5 * sum[ yi - vi ] ^ 2
    """
    return y - v

  def train(self, training_data: np.array, test_data: np.array = None,
            learning_rate: float = 0.01, epochs: int = 100,
            mini_batch_size: int = 10):
    """
    Stochastic Gradient Descent (SGD) algorithm.
    """
    for epoch in range(1, epochs + 1):
      random.shuffle(training_data)

      mini_batches = [
        training_data[i: i + mini_batch_size]
        for i in range(0, len(training_data), mini_batch_size)
      ]

      for mini_batch in mini_batches:
        self._train_mini_batch(mini_batch, learning_rate)

      if epoch % max(epochs // 10, 1) == 0:
        loss = self.total_cost(training_data)
        acc = self.evaluate(test_data) / len(test_data)
        print(f"[{epoch} / {epochs} epoch] loss = {loss:.5} acc = {acc:.3}")

  def _train_mini_batch(self, mini_batch: list, learning_rate: float) -> None:
    n = len(mini_batch)
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    for (x, v) in mini_batch:
      delta_nabla_w, delta_nabla_b = self.backprop(x, v)
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

    self.weights = [w - learning_rate * (nw / n)
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - learning_rate * (nb / n)
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

    nabla_a[-1] = Network.cost_derivative(activations[-1], v)

    for l in range(self.num_layers - 2, -1, -1):
      t = nabla_a[l + 1] * dsigmoid(preactivations[l + 1])

      nabla_w[l] = np.dot(t, activations[l].T)
      nabla_b[l] = t
      nabla_a[l] = np.dot(self.weights[l].T, t)

    return nabla_w, nabla_b

  def evaluate(self, data):
    results = [(np.argmax(self.feedforward(x)), np.argmax(v))
                for (x, v) in data]
    return sum(int(x == y) for (x, y) in results)

  def total_cost(self, data: np.array) -> float:
    n = len(data)
    cost = 0.
    for (x, v) in data:
      y = self.feedforward(x)
      cost += 0.5 * sum((y - v) ** 2)[0]
    return cost / n

def main():
  #np.random.seed(0)
  net = Network([784, 30, 10])

  training_data, test_data = mnist_load()
  training_data = training_data[:2000]

  net.train(training_data, test_data, learning_rate=2, mini_batch_size=20,
            epochs=30)

if __name__ == "__main__":
  main()

