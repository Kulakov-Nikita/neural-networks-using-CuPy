import cupy as cp

from layers.abs_layer import Layer


class RNN(Layer):
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        limit = cp.sqrt(6 / (input_size + hidden_size))
        self.Wxh = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.Whh = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.bias = cp.zeros(hidden_size, dtype=cp.float32)

        self.inp = None
        self.h_states = None
        self.grad_Wxh = None
        self.grad_Whh = None
        self.grad_bias = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = x.shape
        self.inp = x
        dtype = x.dtype
        self.h_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size), dtype=dtype)

        h_t = cp.zeros((batch_size, self.hidden_size), dtype=dtype)
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = cp.tanh(x_t @ self.Wxh + h_t @ self.Whh + self.bias)
            self.h_states[:, t + 1, :] = h_t

        outputs = self.h_states[:, 1:, :]
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = self.inp.shape
        grad_x = cp.zeros_like(self.inp)
        self.grad_Wxh = cp.zeros_like(self.Wxh)
        self.grad_Whh = cp.zeros_like(self.Whh)
        self.grad_bias = cp.zeros_like(self.bias)

        if self.return_sequences:
            grad_h = grad_out
        else:
            grad_h = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=self.inp.dtype)
            grad_h[:, -1, :] = grad_out

        grad_next = cp.zeros((batch_size, self.hidden_size), dtype=self.inp.dtype)
        for t in range(seq_len - 1, -1, -1):
            dh = grad_h[:, t, :] + grad_next
            h_t = self.h_states[:, t + 1, :]
            dtanh = dh * (1 - h_t ** 2)

            self.grad_Wxh += self.inp[:, t, :].T @ dtanh
            self.grad_Whh += self.h_states[:, t, :].T @ dtanh
            self.grad_bias += cp.sum(dtanh, axis=0)

            grad_x[:, t, :] = dtanh @ self.Wxh.T
            grad_next = dtanh @ self.Whh.T

        return grad_x

    def update(self, lr: float, l2: float):
        self.Wxh -= lr * (self.grad_Wxh + l2 * self.Wxh)
        self.Whh -= lr * (self.grad_Whh + l2 * self.Whh)
        self.bias -= lr * self.grad_bias