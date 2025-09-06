import cupy as cp
from cupy._core import ndarray


from layers.abs_layer import Layer


class Dense(Layer):
    def __init__(self, size_in: int, size_out: int):
        limit = cp.sqrt(6 / (size_in + size_out))
        self.weights = cp.random.uniform(-limit, limit, (size_in, size_out))
        self.bias = cp.random.uniform(-limit, limit, size_out)

        self.inp = None
        self.out = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        return x @ self.weights + self.bias

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        self.grad_weights = self.inp.T @ grad_out
        self.grad_bias = cp.sum(grad_out, axis=0)
        grad_in = grad_out @ self.weights.T
        return grad_in

    def update(self, lr: float, l2: float):
        self.weights -= lr * (self.grad_weights + l2 * self.weights)
        self.bias -= lr * self.grad_bias

    

if __name__ == '__main__':
    x = cp.random.rand(100, 10)
    w = cp.random.rand(10, 2)
    b = cp.random.rand(100, 2)

    y = x @ w + b

    print(x.shape, y.shape)

    dense = Dense(x.shape[1], y.shape[1])
