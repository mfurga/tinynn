#!/usr/bin/env python3

from typing import Callable

import pickle
import random
import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def dsigmoid(z):
  return sigmoid(z) * (1. - sigmoid(z))

def softmax(z):
  ezp = np.exp(z - np.max(z))
  return ezp / np.sum(ezp)

class QuadraticCost:
  @staticmethod
  def fn(y: np.array, v: np.array) -> float:
    return 0.5 * np.linalg.norm(y - v) ** 2

  @staticmethod
  def delta(z: np.array, y: np.array, v: np.array) -> np.array:
    return (y - v) * dsigmoid(z)

class CrossEntropyCost:
  @staticmethod
  def fn(y: np.array, v: np.array) -> float:
    return -1 * np.sum(v * np.log(y))

  @staticmethod
  def delta(z: np.array, y: np.array, v: np.array) -> np.array:
    return y - v

class Network:
  def __init__(self, size: tuple, cost = QuadraticCost):
    self.size = size
    self.num_layers = len(size)
    self.cost = cost

    # Weights & biases initialization.
    self.weights = [
      np.random.randn(y, x) for (x, y) in zip(size[:-1], size[1:])
    ]
    self.biases = [np.random.randn(x, 1) for x in size[1:]]

  def feedforward(self, a: np.array) -> np.array:
    for w, b in zip(self.weights, self.biases):
      a = sigmoid(w.dot(a) + b)
    return a

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
    preactivations = [None]
    activations = [x]
    a = x

    for w, b in zip(self.weights, self.biases):
      z = w.dot(a) + b
      preactivations.append(z)
      a = sigmoid(z)
      activations.append(a)

    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    # delta[l + 1]
    delta = self.cost.delta(preactivations[-1], activations[-1], v)

    nabla_w[-1] = np.dot(delta, activations[-2].T)
    nabla_b[-1] = delta

    for l in range(self.num_layers - 3, -1, -1):
      delta = np.dot(self.weights[l + 1].T, delta) * \
          dsigmoid(preactivations[l + 1])

      nabla_w[l] = np.dot(delta, activations[l].T)
      nabla_b[l] = delta

    return nabla_w, nabla_b

  def evaluate(self, data):
    results = [(np.argmax(self.feedforward(x)), np.argmax(v))
                for (x, v) in data]
    return sum(int(x == y) for (x, y) in results)

  def total_cost(self, data: np.array) -> float:
    cost = 0.
    for (x, v) in data:
      y = self.feedforward(x)
      cost += self.cost.fn(y, v)
    return cost / len(data)

def save(net: Network):
  with open("weights.pkl", "wb") as f:
    pickle.dump(net.weights, f, pickle.HIGHEST_PROTOCOL)

  with open("biases.pkl", "wb") as f:
    pickle.dump(net.biases, f, pickle.HIGHEST_PROTOCOL)

def load() -> Network:
  net = Network([784, 60, 10], cost=CrossEntropyCost)
  with open("weights.pkl", "rb") as f:
    net.weights = pickle.load(f)

  with open("biases.pkl", "rb") as f:
    net.biases = pickle.load(f)

  return net

def main():
  #np.random.seed(0)
  #random.seed(0)

  net = Network([3, 2, 1], cost=CrossEntropyCost)
  x = np.array([1, 1, 1]).reshape(-1, 1)
  y = net.feedforward(x)
  print(y)

if __name__ == "__main__":
  main()

