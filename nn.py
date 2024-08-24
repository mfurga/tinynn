#!/usr/bin/env python3

from typing import Callable, List

import sys
import json
import pickle
import random
import numpy as np

from datasets import mnist
from utils import load_mnist_digit, show_mnist_digit


class Sigmoid:
  @staticmethod
  def fn(z: np.array) -> np.array:
    return 1. / (1. + np.exp(-z))

  @staticmethod
  def dfn(z: np.array) -> np.array:
    return Sigmoid.fn(z) * (1. - Sigmoid.fn(z))


class Softmax:
  @staticmethod
  def fn(z: np.array) -> np.array:
    ezp = np.exp(z - np.max(z))
    return ezp / np.sum(ezp)

  @staticmethod
  def dfn(z: np.array) -> np.array:
    raise Exception("Not implemented")


class QuadraticCost:
  @staticmethod
  def fn(y: np.array, v: np.array) -> float:
    return 0.5 * np.linalg.norm(y - v) ** 2

  @staticmethod
  def delta(z: np.array, y: np.array, v: np.array) -> np.array:
    return (y - v) * Sigmoid.dfn(z)


class CrossEntropyCost:
  @staticmethod
  def fn(y: np.array, v: np.array) -> float:
    return -1 * np.sum(v * np.log(y))

  @staticmethod
  def delta(z: np.array, y: np.array, v: np.array) -> np.array:
    # NOTE: Activation of the last layer has to be the softmax function.
    return y - v


class Layer:
  def __init__(self, size_in: int, size_out: int, activation = Sigmoid):
    self.weights = np.random.randn(size_out, size_in)
    self.bias = np.random.randn(size_out, 1)
    self.activation = activation


class Network:
  def __init__(self, layers: List[Layer], cost = QuadraticCost):
    self.layers = layers
    self.num_layers = len(layers) + 1
    self.cost = cost

  def feedforward(self, a: np.array) -> np.array:
    for l in self.layers:
      a = l.activation.fn(l.weights.dot(a) + l.bias)
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
    nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
    nabla_b = [np.zeros(l.bias.shape) for l in self.layers]

    for (x, v) in mini_batch:
      delta_nabla_w, delta_nabla_b = self.backprop(x, v)
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

    for i, l in enumerate(self.layers):
      l.weights = l.weights - learning_rate * (nabla_w[i] / n)
      l.bias = l.bias - learning_rate * (nabla_b[i] / n)

  def backprop(self, x: np.array, v: np.array):
    preactivations = [None]
    activations = [x]
    a = x

    for l in self.layers:
      z = l.weights.dot(a) + l.bias
      preactivations.append(z)
      a = l.activation.fn(z)
      activations.append(a)

    nabla_w = [np.zeros(l.weights.shape) for l in self.layers]
    nabla_b = [np.zeros(l.bias.shape) for l in self.layers]

    # delta[l + 1]
    delta = self.cost.delta(preactivations[-1], activations[-1], v)

    nabla_w[-1] = np.dot(delta, activations[-2].T)
    nabla_b[-1] = delta

    for l in range(self.num_layers - 3, -1, -1):
      delta = np.dot(self.layers[l + 1].weights.T, delta) * \
          self.layers[l].activation.dfn(preactivations[l + 1])

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

  @staticmethod
  def load(fn: str):
    with open(fn, "r") as f:
      data = json.load(f)

    cost = getattr(sys.modules[__name__], data["cost"])
    layers = []

    for l in data["layers"]:
      activation = getattr(sys.modules[__name__], l["activation"])
      layer = Layer(1, 1, activation)
      layer.weights = np.array(l["weights"])
      layer.bias = np.array(l["bias"])
      layers.append(layer)

    return Network(layers, cost)

  def save(self, fn: str) -> None:
    layers = []

    for l in self.layers:
      layers.append({
        "weights": l.weights.tolist(),
        "bias": l.bias.tolist(),
        "activation": l.activation.__name__
      })

    data = {
      "cost": self.cost.__name__,
      "layers": layers
    }

    with open(fn, "w") as f:
      json.dump(data, f)

def main():
  #np.random.seed(0)
  #random.seed(0)

  train = False

  if train:
    training_data, test_data = mnist.load("data/")

    net = Network([
      Layer(784, 100),
      Layer(100, 10, activation=Softmax)
    ], cost=CrossEntropyCost)
    net.train(training_data, test_data[:5000], learning_rate=2.5,
              epochs=20, mini_batch_size=50)
    net.save("net3.json")
  else:
    net = Network.load("net2.json")

  x = load_mnist_digit("data/dig.bmp")
  #show_mnist_digit(x)
  y = net.feedforward(x)
  print(y)
  print(np.argmax(y))
  print(np.sum(y))

if __name__ == "__main__":
  main()

