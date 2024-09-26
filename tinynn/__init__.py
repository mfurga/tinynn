from tinynn.tensor import Tensor

def sigmoid(x: Tensor) -> Tensor:
  return 1. / (1. + (-x).exp())

