import unittest
import numpy as np

from tinynn import Tensor


class TestMean(unittest.TestCase):

  def test_mean_1d(self):
    x = Tensor([3, 1, 8], requires_grad=True)

    y = x.mean()
    assert y.equals(Tensor(4))

    y.backward()
    assert x.grad.equals(Tensor([1/3, 1/3, 1/3]))

  def test_mean_2d(self):
    x = Tensor([[3, 1, 8], [-4, 4, 6]], requires_grad=True)

    y = x.mean()
    assert y.equals(Tensor(3))

    y.backward()
    assert x.grad.equals(Tensor([[1/6, 1/6, 1/6], [1/6, 1/6, 1/6]]))

  def test_mean_2d_axis0(self):
    x = Tensor([[3, 1, 8], [-4, 4, 6]], requires_grad=True)

    y = x.mean(axis=0)
    assert y.equals(Tensor([-0.5, 2.5, 7.]))

    y.backward()
    assert x.grad.equals(Tensor([[1/2, 1/2, 1/2], [1/2, 1/2, 1/2]]))

  def test_mean_2d_axis1(self):
    x = Tensor([[3, 1, 8], [-4, 4, 6]], requires_grad=True)

    y = x.mean(axis=1)
    assert y.equals(Tensor([4, 2]))

    y.backward()
    assert x.grad.equals(Tensor([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]))

  def test_mean_3d_axis0(self):
    x = Tensor([
      [[3, 1, 8], [-4, 4, 6]],
      [[4, 3, -4], [9, 1, -8]]
    ], requires_grad=True)

    y = x.mean(axis=0)
    assert y.equals(Tensor([[3.5, 2, 2], [2.5, 2.5, -1]]))

    y.backward()
    assert x.grad.equals(Tensor([
      [[1/2, 1/2, 1/2], [1/2, 1/2, 1/2]],
      [[1/2, 1/2, 1/2], [1/2, 1/2, 1/2]]
    ]))


if __name__ == "__main__":
  unittest.main()

