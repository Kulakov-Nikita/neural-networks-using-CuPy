import cupy as cp
from cupy.lib.stride_tricks import as_strided
from layers.abs_layer import Layer


def im2col(x, kernel_size, stride, padding):
    batch_size, channels, H, W = x.shape

    if padding > 0:
        x = cp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_padded, W_padded = x.shape[2], x.shape[3]

    H_out = (H_padded - kernel_size) // stride + 1
    W_out = (W_padded - kernel_size) // stride + 1

    shape = (batch_size, channels, kernel_size, kernel_size, H_out, W_out)
    stride_b, stride_c, stride_h, stride_w = x.strides
    strides = (stride_b, stride_c, stride_h, stride_w, stride_h * stride, stride_w * stride)

    patches = as_strided(x, shape=shape, strides=strides)
    cols = patches.reshape(batch_size, channels * kernel_size * kernel_size, H_out * W_out)
    return cols


def col2im(cols, x_shape, kernel_size, stride, padding):
    batch_size, channels, H, W = x_shape

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - kernel_size) // stride + 1
    W_out = (W_padded - kernel_size) // stride + 1

    x_padded = cp.zeros((batch_size, channels, H_padded, W_padded), dtype=cols.dtype)

    stride_b, stride_c, stride_h, stride_w = x_padded.strides

    view_shape = (batch_size, channels, kernel_size, kernel_size, H_out, W_out)
    view_strides = (stride_b, stride_c, stride_h, stride_w, stride_h * stride, stride_w * stride)

    view = as_strided(x_padded, shape=view_shape, strides=view_strides)
    view += cols.reshape(view_shape)

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    else:
        return x_padded


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, has_activation=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_activation = has_activation

        limit = cp.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.weights = cp.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = cp.zeros(out_channels)

        self.inp = None
        self.out = None
        self.cols = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        batch_size, channels, H, W = x.shape


        self.cols = im2col(x, self.kernel_size, self.stride, self.padding)  # (batch_size, channels*kernel_size*kernel_size, H_out*W_out)
        W_col = self.weights.reshape(self.out_channels, -1)  # (filters, channels*kernel_size*kernel_size)

        out_temp = cp.tensordot(self.cols, W_col.T, axes=([1], [0]))  # (batch_size, H_out*W_out, filters)
        out_temp = out_temp.transpose(0, 2, 1)  # (batch_size, filters, H_out*W_out)

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out_temp.reshape(batch_size, self.out_channels, H_out, W_out)
        out = out + self.bias[None, :, None, None]

        if self.has_activation:
            self.out = 1 / (1 + cp.exp(-out))
        else:
            self.out = out
        return self.out

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, channels, H_out, W_out = grad_out.shape
        N = H_out * W_out

        if self.has_activation:
            grad_out = grad_out * (self.out * (1 - self.out))

        grad_out_reshaped = grad_out.reshape(batch_size, self.out_channels, -1)  # (batch_size, filters, H_out*W_out)

        self.grad_bias = cp.sum(grad_out_reshaped, axis=(0, 2))  # (filters,)

        self.grad_weights = cp.einsum('bon,bcn->oc', grad_out_reshaped, self.cols)  # (out, Ck)
        self.grad_weights = self.grad_weights.reshape(self.weights.shape)  # (out, in, k, k)

        W_col = self.weights.reshape(self.out_channels, -1)  # (filters, Ck)
        grad_cols = cp.einsum('bon,oc->bcn', grad_out_reshaped, W_col)  # (B, channels*kernel_size*kernel_size, H_out*W_out)

        grad_in = col2im(grad_cols, self.inp.shape, self.kernel_size, self.stride, self.padding)  # (batch_size, channels, H, W)
        return grad_in

    def update(self, lr: float, l2: float):
        self.weights -= lr * (self.grad_weights + l2 * self.weights)
        self.bias -= lr * self.grad_bias


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.shape)
    
    def update(self, lr: float, l2: float):
        pass
    

class MaxPool2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.inp = None
        self.argmax = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        batch_size, channels, H, W = x.shape

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        out = cp.zeros((batch_size, channels, H_out, W_out))
        self.argmax = cp.zeros_like(out, dtype=cp.int32)

        for i in range(H_out):
            for j in range(W_out):
                h_start, h_end = i * self.stride, i * self.stride + self.kernel_size
                w_start, w_end = j * self.stride, j * self.stride + self.kernel_size
                window = x[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(batch_size, channels, -1)
                max_idx = cp.argmax(window_reshaped, axis=2)
                max_val = cp.max(window_reshaped, axis=2)
                out[:, :, i, j] = max_val
                self.argmax[:, :, i, j] = max_idx

        return out

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, channels, H, W = self.inp.shape
        k, s = self.kernel_size, self.stride
        _, _, H_out, W_out = grad_out.shape

        grad_in = cp.zeros_like(self.inp)

        for i in range(H_out):
            for j in range(W_out):
                h_start, h_end = i * s, i * s + k
                w_start, w_end = j * s, j * s + k

                window_grad = cp.zeros((batch_size, channels, k * k))
                idx = self.argmax[:, :, i, j]
                window_grad = window_grad.reshape(batch_size, channels, -1)
                batch_idx = cp.arange(batch_size)[:, None]
                chan_idx = cp.arange(channels)[None, :]
                window_grad[batch_idx, chan_idx, idx] = grad_out[:, :, i, j]
                grad_in[:, :, h_start:h_end, w_start:w_end] += window_grad.reshape(batch_size, channels, k, k)

        return grad_in


if __name__ == "__main__":
    x = cp.random.randn(8, 3, 32, 32)
    conv = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
    y = conv.forward(x)
    print("Forward:", y.shape)
    grad = cp.random.randn(*y.shape)
    dx = conv.backward(grad)
    print("Backward:", dx.shape)
