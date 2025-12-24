import cupy as cp

from layers.abs_layer import Layer


class TokenEmbedding(Layer):
    def __init__(self, vocab_size: int, d_model: int):
        limit = cp.sqrt(cp.float32(6.0) / cp.float32(vocab_size + d_model))
        self.weights = cp.random.uniform(-limit, limit, (vocab_size, d_model)).astype(cp.float32)
        self.inp = None
        self.grad_weights = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x.astype(cp.int32)
        return self.weights[self.inp]

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        self.grad_weights = cp.zeros_like(self.weights)
        cp.add.at(self.grad_weights, self.inp, grad_out)
        return cp.zeros_like(self.inp, dtype=grad_out.dtype)

    def update(self, lr: float, l2: float):
        self.weights -= lr * (self.grad_weights + l2 * self.weights)


class PositionalEncoding(Layer):
    def __init__(self, max_len: int, d_model: int):
        position = cp.arange(max_len, dtype=cp.float32)[:, None]
        div_term = cp.exp(cp.arange(0, d_model, 2, dtype=cp.float32) * (-cp.log(cp.float32(10000.0)) / d_model))
        pe = cp.zeros((max_len, d_model), dtype=cp.float32)
        pe[:, 0::2] = cp.sin(position * div_term)
        pe[:, 1::2] = cp.cos(position * div_term)
        self.encoding = pe
        self.seq_len = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.seq_len = x.shape[1]
        if self.seq_len > self.encoding.shape[0]:
            raise ValueError(f"PositionalEncoding max_len={self.encoding.shape[0]} is smaller than sequence length {self.seq_len}")
        return x + self.encoding[: self.seq_len]

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        return grad_out

    def update(self, lr: float, l2: float):
        pass


class LayerNorm(Layer):
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = cp.ones(d_model, dtype=cp.float32)
        self.beta = cp.zeros(d_model, dtype=cp.float32)
        self.eps = cp.float32(eps)
        self.inp = None
        self.mean = None
        self.var = None
        self.x_hat = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        self.mean = cp.mean(x, axis=-1, keepdims=True)
        self.var = cp.mean((x - self.mean) ** 2, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        m = self.inp.shape[-1]
        self.grad_gamma = cp.sum(grad_out * self.x_hat, axis=(0, 1))
        self.grad_beta = cp.sum(grad_out, axis=(0, 1))

        dx_hat = grad_out * self.gamma
        dvar = cp.sum(dx_hat * (self.inp - self.mean) * (-0.5) * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = cp.sum(dx_hat * (-1.0) / cp.sqrt(self.var + self.eps), axis=-1, keepdims=True) + dvar * cp.sum(-2.0 * (self.inp - self.mean), axis=-1, keepdims=True) / m

        grad_in = dx_hat / cp.sqrt(self.var + self.eps) + dvar * 2.0 * (self.inp - self.mean) / m + dmean / m
        return grad_in

    def update(self, lr: float, l2: float):
        self.gamma -= lr * (self.grad_gamma + l2 * self.gamma)
        self.beta -= lr * self.grad_beta


class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        limit = cp.sqrt(cp.float32(6.0) / cp.float32(4 * d_model))
        self.W_q = cp.random.uniform(-limit, limit, (d_model, d_model)).astype(cp.float32)
        self.W_k = cp.random.uniform(-limit, limit, (d_model, d_model)).astype(cp.float32)
        self.W_v = cp.random.uniform(-limit, limit, (d_model, d_model)).astype(cp.float32)
        self.W_o = cp.random.uniform(-limit, limit, (d_model, d_model)).astype(cp.float32)

        self.inp = None
        self.Q_h = None
        self.K_h = None
        self.V_h = None
        self.attn = None
        self.context = None

        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None

        self.pad_mask = None

    def set_pad_mask(self, mask: cp.ndarray):
        self.pad_mask = mask.astype(cp.bool_)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        self.Q_h = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.K_h = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        self.V_h = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = cp.sqrt(cp.float32(self.head_dim))
        scores = cp.matmul(self.Q_h, self.K_h.transpose(0, 1, 3, 2)) / scale

        if self.pad_mask is not None:
            m = self.pad_mask[:, None, None, :]
            scores = cp.where(m, scores, cp.float32(-1e9))

        scores = scores - cp.max(scores, axis=-1, keepdims=True)
        exp_scores = cp.exp(scores)
        self.attn = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

        context_h = cp.matmul(self.attn, self.V_h)
        self.context = context_h.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.context @ self.W_o

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = grad_out.shape
        scale = cp.sqrt(cp.float32(self.head_dim))

        self.grad_W_o = self.context.reshape(-1, self.d_model).T @ grad_out.reshape(-1, self.d_model)
        grad_context = grad_out @ self.W_o.T
        grad_context_h = grad_context.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        grad_attn = cp.matmul(grad_context_h, self.V_h.transpose(0, 1, 3, 2))
        grad_V_h = cp.matmul(self.attn.transpose(0, 1, 3, 2), grad_context_h)

        grad_scores = self.attn * (grad_attn - cp.sum(grad_attn * self.attn, axis=-1, keepdims=True))
        grad_scores = grad_scores / scale

        grad_Q_h = cp.matmul(grad_scores, self.K_h)
        grad_K_h = cp.matmul(grad_scores.transpose(0, 1, 3, 2), self.Q_h)

        grad_Q = grad_Q_h.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_K = grad_K_h.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_V = grad_V_h.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        inp_flat = self.inp.reshape(-1, self.d_model)
        self.grad_W_q = inp_flat.T @ grad_Q.reshape(-1, self.d_model)
        self.grad_W_k = inp_flat.T @ grad_K.reshape(-1, self.d_model)
        self.grad_W_v = inp_flat.T @ grad_V.reshape(-1, self.d_model)

        grad_in = grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T
        return grad_in

    def update(self, lr: float, l2: float):
        self.W_q -= lr * (self.grad_W_q + l2 * self.W_q)
        self.W_k -= lr * (self.grad_W_k + l2 * self.W_k)
        self.W_v -= lr * (self.grad_W_v + l2 * self.W_v)
        self.W_o -= lr * (self.grad_W_o + l2 * self.W_o)


class PositionwiseFeedForward(Layer):
    def __init__(self, d_model: int, d_hidden: int):
        limit1 = cp.sqrt(cp.float32(6.0) / cp.float32(d_model + d_hidden))
        limit2 = cp.sqrt(cp.float32(6.0) / cp.float32(d_hidden + d_model))
        self.W1 = cp.random.uniform(-limit1, limit1, (d_model, d_hidden)).astype(cp.float32)
        self.b1 = cp.zeros(d_hidden, dtype=cp.float32)
        self.W2 = cp.random.uniform(-limit2, limit2, (d_hidden, d_model)).astype(cp.float32)
        self.b2 = cp.zeros(d_model, dtype=cp.float32)

        self.inp = None
        self.hidden = None
        self.hidden_relu = None

        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        self.hidden = x @ self.W1 + self.b1
        self.hidden_relu = cp.maximum(self.hidden, 0)
        return self.hidden_relu @ self.W2 + self.b2

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        grad_hidden_relu = grad_out @ self.W2.T
        grad_hidden = grad_hidden_relu * (self.hidden > 0)

        self.grad_W2 = self.hidden_relu.reshape(-1, self.hidden_relu.shape[-1]).T @ grad_out.reshape(-1, grad_out.shape[-1])
        self.grad_b2 = cp.sum(grad_out, axis=(0, 1))

        self.grad_W1 = self.inp.reshape(-1, self.inp.shape[-1]).T @ grad_hidden.reshape(-1, grad_hidden.shape[-1])
        self.grad_b1 = cp.sum(grad_hidden, axis=(0, 1))

        grad_in = grad_hidden @ self.W1.T
        return grad_in

    def update(self, lr: float, l2: float):
        self.W1 -= lr * (self.grad_W1 + l2 * self.W1)
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * (self.grad_W2 + l2 * self.W2)
        self.b2 -= lr * self.grad_b2


class TransformerEncoderBlock(Layer):
    def __init__(self, d_model: int, num_heads: int, ff_hidden: int):
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, ff_hidden)
        self.norm2 = LayerNorm(d_model)
        self.inp = None
        self.attn_out = None
        self.attn_res = None
        self.norm1_out = None
        self.ff_out = None
        self.ff_res = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.inp = x
        self.attn_out = self.self_attn.forward(x)
        self.attn_res = x + self.attn_out
        self.norm1_out = self.norm1.forward(self.attn_res)

        self.ff_out = self.ff.forward(self.norm1_out)
        self.ff_res = self.norm1_out + self.ff_out
        return self.norm2.forward(self.ff_res)

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        grad_ff_res = self.norm2.backward(grad_out)

        grad_ff_out = grad_ff_res
        grad_norm1_out = grad_ff_res

        grad_norm1_inp = self.ff.backward(grad_ff_out)
        grad_norm1_total = grad_norm1_out + grad_norm1_inp

        grad_attn_res = self.norm1.backward(grad_norm1_total)

        grad_attn_out = grad_attn_res
        grad_in_direct = grad_attn_res

        grad_self_attn_in = self.self_attn.backward(grad_attn_out)

        return grad_in_direct + grad_self_attn_in

    def update(self, lr: float, l2: float):
        self.self_attn.update(lr, l2)
        self.norm1.update(lr, l2)
        self.ff.update(lr, l2)
        self.norm2.update(lr, l2)


class MeanPooling(Layer):
    def __init__(self):
        self.seq_len = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        self.seq_len = x.shape[1]
        return cp.mean(x, axis=1)

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        expanded = cp.repeat(grad_out[:, None, :], self.seq_len, axis=1)
        return expanded / cp.float32(self.seq_len)

    def update(self, lr: float, l2: float):
        pass
