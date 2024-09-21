import unittest
import numpy as np

from tinynn import Tensor


class TestGradDot(unittest.TestCase):

  def test_dot_1d_1d(self):
    x = Tensor([1, 9, 7], requires_grad=True)
    y = Tensor([4, 10, 8], requires_grad=True)

    z = x.dot(y)
    assert z.equal(Tensor(150))

    z.backward()
    assert x.grad.equal(Tensor([4, 10, 8]))
    assert y.grad.equal(Tensor([1, 9, 7]))

  def test_dot_2d_1d(self):
    x = Tensor([
      [1, -12,  49, -42],
      [0,   9,  -3,   0],
      [41,  0, -13,   3]
    ], requires_grad=True)

    y = Tensor([-5, 3, 14, -25], requires_grad=True)

    z = x.dot(y)
    assert z.equal(Tensor([1695, -15, -462]))

    z.backward()
    assert x.grad.equal(Tensor([
      [-5, 3, 14, -25],
      [-5, 3, 14, -25],
      [-5, 3, 14, -25]
    ]))
    assert y.grad.equal(Tensor([42, -3, 33, -39]))

  def test_dot_1d_2d(self):
    x = Tensor([-5, 3, 14, -25], requires_grad=True)

    y = Tensor([
      [1, -12,  49],
      [0,   9,  -3],
      [41,  0, -13],
      [2,  -3,  13]
    ], requires_grad=True)

    z = x.dot(y)
    assert z.equal(Tensor([519, 162, -761]))

    z.backward()
    assert x.grad.equal(Tensor([38, 6, 28, 12]))
    assert y.grad.equal(Tensor([
      [ -5,  -5,  -5],
      [  3,   3,   3],
      [ 14,  14,  14],
      [-25, -25, -25]
    ]))

  def test_dot_2d_2d(self):
    x = Tensor([
      [7, 10, 4],
      [6, -9, 3]
    ], requires_grad=True)

    y = Tensor([
      [8 ,  9,  -4, 13],
      [4 , -1,   6, 10],
      [10, 47, -13,  1]
    ], requires_grad=True)

    z = x.dot(y)
    assert z.equal(Tensor([
      [136, 241,  -20, 195],
      [ 42, 204, -117,  -9]
    ]))

    z.backward()
    assert x.grad.equal(Tensor([
      [26, 19, 45],
      [26, 19, 45]
    ]))
    assert y.grad.equal(Tensor([
      [13, 13, 13, 13],
      [ 1,  1,  1,  1],
      [ 7,  7,  7,  7]
    ]))


if __name__ == "__main__":
  unittest.main()

