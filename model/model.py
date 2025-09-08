from typing import List, Callable

import cupy as cp
from tqdm import trange

from layers.abs_layer import Layer
from layers.dense import Dense

epsilon = 1e-8

def cross_entropy(y_pred,y_true):
    return -cp.mean(cp.sum(y_true * cp.log(y_pred + epsilon), axis=1))

class NNet:
    def __init__(self, layers: List[Layer], loss_fn: Callable):
        self.layers: List[Layer] = layers
        self.loss_fn: Callable = loss_fn

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad: cp.ndarray):
        grad_out = grad
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
            
    def update(self, lr, l2):
        for layer in self.layers:
            layer.update(lr, l2)

    def fit(self, X, y, num_epochs=100, lr=cp.float32(0.001), l2=cp.float32(0.001), batch_size: int | None = None, shuffle: bool = True):
        n_samples = len(X)
        if batch_size is None or batch_size <= 0:
            batch_size = n_samples

        loss_story = cp.zeros(num_epochs)

        pbar = trange(num_epochs, desc="Training", leave=True)
        for epoch in pbar:
            indices = cp.random.permutation(n_samples) if shuffle else cp.arange(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_b, y_b = X[batch_idx], y[batch_idx]

                y_prob = self.forward(X_b)
                batch_loss = self.loss_fn(y_prob, y_b)
                epoch_loss += batch_loss * (end - start)

                error = (y_prob - y_b) / (end - start)
                self.backward(error)
                self.update(lr, l2)

            loss_story[epoch] = epoch_loss / n_samples
            pbar.set_postfix(loss=float(loss_story[epoch]))

        return loss_story

    def predict(self, X, batch_size: int = 256):
        n_samples = len(X)
        if batch_size is None or batch_size <= 0:
            batch_size = n_samples

        outputs = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_b = X[start:end]
            y_b = self.forward(X_b)
            outputs.append(y_b)

        return cp.concatenate(outputs, axis=0)
    

if __name__ == '__main__':
    x = cp.random.rand(1000, 10)
    w = cp.random.rand(10, 2)
    b = cp.random.rand(100, 2)
    
    y = x @ w

    dense1 = Dense(x.shape[1], 20)
    dense2 = Dense(20, y.shape[1])

    model = NNet([dense1, dense2], cross_entropy)
    loss = model.fit(x, y)
    import matplotlib.pyplot as plt
    plt.plot(range(len(loss)), loss)
