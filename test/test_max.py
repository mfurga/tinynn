import unittest
import numpy as np

from tinynn import Tensor


class TestMax(unittest.TestCase):

  def test_max_1d(self):
    x = Tensor([1, 9, 7], requires_grad=True)

    y = x.max()
    assert y.equal(Tensor(9))

    y.backward()
    assert x.grad.equal(Tensor([0, 1, 0]))

  def test_max_1d_with_dups(self):
    x = Tensor([10, -3, 1, 10], requires_grad=True)

    y = x.max()
    assert y.equal(Tensor(10))

    y.backward()
    assert x.grad.equal(Tensor([0.5, 0, 0, 0.5]))

  def test_max_2d(self):
    x = Tensor([[3, 1, -4], [-6, 12, 2]], requires_grad=True)

    y = x.max()
    assert y.equal(Tensor(12))

    y.backward()
    assert x.grad.equal(Tensor([[0, 0, 0], [0, 1, 0]]))

  def test_max_2d_with_dups(self):
    x = Tensor([[12, 12, -4], [-6, 12, 2]], requires_grad=True)

    y = x.max()
    assert y.equal(Tensor(12))

    y.backward()
    assert x.grad.equal(Tensor([[1/3, 1/3, 0], [0, 1/3, 0]]))

  def test_max_2d_axis0(self):
    x = Tensor([[4, 2, -5], [-6, 12, 1]], requires_grad=True)

    y = x.max(axis=0)
    assert y.equal(Tensor([4, 12, 1]))

    y.backward()
    assert x.grad.equal(Tensor([[1, 0, 0], [0, 1, 1]]))

  def test_max_2d_axis0_with_dups(self):
    x = Tensor([[-6, 9, 3], [-6, 12, 3]], requires_grad=True)

    y = x.max(axis=0)
    assert y.equal(Tensor([-6, 12, 3]))

    y.backward()
    assert x.grad.equal(Tensor([[0.5, 0, 0.5], [0.5, 1, 0.5]]))

  def test_max_2d_axis1(self):
    x = Tensor([[4, 2, -5], [-6, 12, 1]], requires_grad=True)

    y = x.max(axis=1)
    assert y.equal(Tensor([4, 12]))

    y.backward()
    assert x.grad.equal(Tensor([[1, 0, 0], [0, 1, 0]]))

  def test_max_2d_axis1_with_dups(self):
    x = Tensor([[-6, 9, 9], [12, -3, 12]], requires_grad=True)

    y = x.max(axis=1)
    assert y.equal(Tensor([9, 12]))

    y.backward()
    assert x.grad.equal(Tensor([[0, 0.5, 0.5], [0.5, 0, 0.5]]))

if __name__ == "__main__":
  unittest.main()

