# TinyNN

Tiny neural network library with automatic differentiation.

### Example usage

```python
from tinynn.nn import Module, Linear, MSELoss
from tinynn.optim import SGD
from tinynn import Tensor

class Network(Module):
  def __init__(self):
    self.l1 = Linear(784, 100)
    self.l2 = Linear(100, 3)

  def forward(self, x: Tensor) -> Tensor:
    r1 = self.l1(x).sigmoid().dropout(0.2)
    r2 = self.l2(r1).softmax()
    return r2

net = Network()
optim = SGD(net.parameters(), lr=2, momentum=0.6)
loss_fn = MSELoss()

X, Y = Tensor.rand(1000, 784), Tensor([0, 1, 0])

Tensor.train()
for epoch in range(20):
  optim.zero_grad()
  loss = loss_fn(net(X), Y)
  loss.backward()
  optim.step()
  print(f"{epoch} {loss}")


```
