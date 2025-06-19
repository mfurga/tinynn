import pytest

from tinynn import Tensor


@pytest.mark.parametrize(
    "x, axis, y, grad",
    [
        ([1, 9, 7], None, 9, [0, 1, 0]),
        ([10, -3, 1, 10], None, 10, [0.5, 0, 0, 0.5]),
        ([[3, 1, -4], [-6, 12, 2]], None, 12, [[0, 0, 0], [0, 1, 0]]),
        ([[12, 12, -4], [-6, 12, 2]], None, 12, [[1 / 3, 1 / 3, 0], [0, 1 / 3, 0]]),
        ([[4, 2, -5], [-6, 12, 1]], 0, [4, 12, 1], [[1, 0, 0], [0, 1, 1]]),
        ([[-6, 9, 3], [-6, 12, 3]], 0, [-6, 12, 3], [[0.5, 0, 0.5], [0.5, 1, 0.5]]),
        ([[4, 2, -5], [-6, 12, 1]], 1, [4, 12], [[1, 0, 0], [0, 1, 0]]),
        ([[-6, 9, 9], [12, -3, 12]], 1, [9, 12], [[0, 0.5, 0.5], [0.5, 0, 0.5]]),
    ],
)
def test_max(x, axis, y, grad):
    x = Tensor(x, requires_grad=True)
    t = x.max(axis=axis)
    assert t.equals(Tensor(y))

    t.backward()
    assert x.grad.equals(Tensor(grad))


@pytest.mark.parametrize(
    "x, axis, y, grad",
    [
        ([3, 1, 8], None, 4, [1 / 3, 1 / 3, 1 / 3]),
        (
            [[3, 1, 8], [-4, 4, 6]],
            None,
            3,
            [[1 / 6, 1 / 6, 1 / 6], [1 / 6, 1 / 6, 1 / 6]],
        ),
        (
            [[3, 1, 8], [-4, 4, 6]],
            0,
            [-0.5, 2.5, 7.0],
            [[1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]],
        ),
        (
            [[3, 1, 8], [-4, 4, 6]],
            1,
            [4, 2],
            [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]],
        ),
        (
            [[[3, 1, 8], [-4, 4, 6]], [[4, 3, -4], [9, 1, -8]]],
            0,
            [[3.5, 2, 2], [2.5, 2.5, -1]],
            [
                [[1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]],
                [[1 / 2, 1 / 2, 1 / 2], [1 / 2, 1 / 2, 1 / 2]],
            ],
        ),
    ],
)
def test_mean(x, axis, y, grad):
    x = Tensor(x, requires_grad=True)
    t = x.mean(axis=axis)
    assert t.equals(Tensor(y))

    t.backward()
    assert x.grad.equals(Tensor(grad))


@pytest.mark.parametrize(
    "x, axis, grad_in, y, grad",
    [
        ([3, 1, 9], None, None, 13, [1, 1, 1]),
        ([[3, 1, 9], [6, 7, -4]], None, None, 22, [[1, 1, 1], [1, 1, 1]]),
        (
            [[3, 1, 9], [6, 7, -4]],
            0,
            [12, -2, 9],
            [9, 8, 5],
            [[12, -2, 9], [12, -2, 9]],
        ),
        ([[3, 1, 9], [6, 7, -4]], 1, [12, -3], [13, 9], [[12, 12, 12], [-3, -3, -3]]),
    ],
)
def test_sum(x, axis, grad_in, y, grad):
    x = Tensor(x, requires_grad=True)
    t = x.sum(axis=axis)
    assert t.equals(Tensor(y))

    if grad_in is None:
        t.backward()
    else:
        t.backward(Tensor(grad_in))

    assert x.grad.equals(Tensor(grad))


@pytest.mark.parametrize(
    "x, y, z, grad_x, grad_y",
    [
        # 1D x 1D
        (
            [1, 9, 7],
            [4, 10, 8],
            150,
            [4, 10, 8],
            [1, 9, 7],
        ),
        # 2D x 1D
        (
            [[1, -12, 49, -42], [0, 9, -3, 0], [41, 0, -13, 3]],
            [-5, 3, 14, -25],
            [1695, -15, -462],
            [[-5, 3, 14, -25], [-5, 3, 14, -25], [-5, 3, 14, -25]],
            [42, -3, 33, -39],
        ),
        # 1D x 2D
        (
            [-5, 3, 14, -25],
            [[1, -12, 49], [0, 9, -3], [41, 0, -13], [2, -3, 13]],
            [519, 162, -761],
            [38, 6, 28, 12],
            [[-5, -5, -5], [3, 3, 3], [14, 14, 14], [-25, -25, -25]],
        ),
        # 2D x 2D
        (
            [[7, 10, 4], [6, -9, 3]],
            [[8, 9, -4, 13], [4, -1, 6, 10], [10, 47, -13, 1]],
            [[136, 241, -20, 195], [42, 204, -117, -9]],
            [[26, 19, 45], [26, 19, 45]],
            [[13, 13, 13, 13], [1, 1, 1, 1], [7, 7, 7, 7]],
        ),
    ],
)
def test_dot(x, y, z, grad_x, grad_y):
    x = Tensor(x, requires_grad=True)
    y = Tensor(y, requires_grad=True)

    t = x.dot(y)
    assert t.equals(Tensor(z))

    t.backward()
    assert x.grad.equals(Tensor(grad_x))
    assert y.grad.equals(Tensor(grad_y))
