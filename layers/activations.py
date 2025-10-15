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


class ReLU(Layer):
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        self.out = cp.maximum(0, x)
        return self.out
    
    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        return grad_out * (self.inp > 0)
    
    def update(self, lr: float, l2: float):
        pass


class SoftMax(Layer):
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        x_shifted = x - cp.max(x,axis=1,keepdims=True)
        return cp.exp(x_shifted)/cp.sum(cp.exp(x_shifted),axis=1, keepdims=True)
    
    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        return grad_out
    
    def update(self, lr: float, l2: float):
        pass
