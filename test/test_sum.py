import unittest
import numpy as np

from tinynn import Tensor


class TestSum(unittest.TestCase):

  def test_sum_1d(self):
    x = Tensor([3, 1, 9], requires_grad=True)

    y = x.sum()
    assert y.equal(Tensor(13))

    y.backward()
    assert x.grad.equal(Tensor([1, 1, 1]))

  def test_sum_2d(self):
    x = Tensor([[3, 1, 9], [6, 7, -4]], requires_grad=True)

    y = x.sum()
    assert y.equal(Tensor(22))

    y.backward()
    assert x.grad.equal(Tensor([[1, 1, 1], [1, 1, 1]]))

  def test_sum_2d_axis0(self):
    x = Tensor([[3, 1, 9], [6, 7, -4]], requires_grad=True)

    y = x.sum(axis=0)
    assert y.equal(Tensor([9, 8, 5]))

    y.backward(Tensor([12, -2, 9]))
    assert x.grad.equal(Tensor([[12, -2, 9], [12, -2, 9]]))

  def test_sum_2d_axis1(self):
    x = Tensor([[3, 1, 9], [6, 7, -4]], requires_grad=True)

    y = x.sum(axis=1)
    assert y.equal(Tensor([13, 9]))

    y.backward(Tensor([12, -3]))
    assert x.grad.equal(Tensor([[12, 12, 12], [-3, -3, -3]]))


if __name__ == "__main__":
  unittest.main()

