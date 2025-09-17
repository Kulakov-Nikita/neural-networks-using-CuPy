import cupy as cp


from layers.abs_layer import Layer


class Sigmoid(Layer):
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.out = 1 / (1 + cp.exp(-x))
        return self.out
    
    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        grad_activation = self.out * (1 - self.out)
        return grad_out * grad_activation
    
    def update(self, lr: float, l2: float):
        pass
