import cupy as cp
from layers.abs_layer import Layer


class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        limit = cp.sqrt(6 / (input_size + hidden_size))

        self.W_i = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_f = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_c = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_o = cp.random.uniform(-limit, limit, (input_size, hidden_size))

        self.U_i = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_f = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_c = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_o = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))

        self.b_i = cp.zeros(hidden_size, dtype=cp.float32)
        self.b_f = cp.zeros(hidden_size, dtype=cp.float32)
        self.b_c = cp.zeros(hidden_size, dtype=cp.float32)
        self.b_o = cp.zeros(hidden_size, dtype=cp.float32)

        self.inp = None
        self.h_states = None
        self.c_states = None
        self.i = None
        self.f = None
        self.c_tilde = None
        self.o = None

        self.grad_W_i = None
        self.grad_W_f = None
        self.grad_W_c = None
        self.grad_W_o = None

        self.grad_U_i = None
        self.grad_U_f = None
        self.grad_U_c = None
        self.grad_U_o = None

        self.grad_b_i = None
        self.grad_b_f = None
        self.grad_b_c = None
        self.grad_b_o = None


    def forward(self, x: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = x.shape
        self.inp = x  # (batch_size, seq_len, input_size)
        dtype = x.dtype

        self.h_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size), dtype=dtype)
        self.c_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size), dtype=dtype)

        self.i = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
        self.f = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
        self.c_tilde = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
        self.o = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)

        for t in range(seq_len):
            x_t = x[:, t, :]                       # (batch_size, input_size)
            h_prev = self.h_states[:, t, :]        # (batch_size, hidden_size)
            c_prev = self.c_states[:, t, :]        # (batch_size, hidden_size)

            i_lin = x_t @ self.W_i + h_prev @ self.U_i + self.b_i
            f_lin = x_t @ self.W_f + h_prev @ self.U_f + self.b_f
            c_lin = x_t @ self.W_c + h_prev @ self.U_c + self.b_c
            o_lin = x_t @ self.W_o + h_prev @ self.U_o + self.b_o

            i_t = 1.0 / (1.0 + cp.exp(-i_lin))   # sigmoid
            f_t = 1.0 / (1.0 + cp.exp(-f_lin))
            c_tilde_t = cp.tanh(c_lin)
            o_t = 1.0 / (1.0 + cp.exp(-o_lin))

            c_t = f_t * c_prev + i_t * c_tilde_t
            h_t = o_t * cp.tanh(c_t)

            self.i[:, t, :] = i_t
            self.f[:, t, :] = f_t
            self.c_tilde[:, t, :] = c_tilde_t
            self.o[:, t, :] = o_t

            self.c_states[:, t + 1, :] = c_t
            self.h_states[:, t + 1, :] = h_t

        outputs = self.h_states[:, 1:, :]
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]           
    

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = self.inp.shape
        dtype = self.inp.dtype

        grad_x = cp.zeros_like(self.inp)

        self.grad_W_i = cp.zeros_like(self.W_i)
        self.grad_W_f = cp.zeros_like(self.W_f)
        self.grad_W_c = cp.zeros_like(self.W_c)
        self.grad_W_o = cp.zeros_like(self.W_o)

        self.grad_U_i = cp.zeros_like(self.U_i)
        self.grad_U_f = cp.zeros_like(self.U_f)
        self.grad_U_c = cp.zeros_like(self.U_c)
        self.grad_U_o = cp.zeros_like(self.U_o)

        self.grad_b_i = cp.zeros_like(self.b_i)
        self.grad_b_f = cp.zeros_like(self.b_f)
        self.grad_b_c = cp.zeros_like(self.b_c)
        self.grad_b_o = cp.zeros_like(self.b_o)

        if self.return_sequences:
            grad_h = grad_out
        else:
            grad_h = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
            grad_h[:, -1, :] = grad_out

        dh_next = cp.zeros((batch_size, self.hidden_size), dtype=dtype)
        dc_next = cp.zeros((batch_size, self.hidden_size), dtype=dtype)

        for t in range(seq_len - 1, -1, -1):
            dh = grad_h[:, t, :] + dh_next

            c_t = self.c_states[:, t + 1, :]
            c_prev = self.c_states[:, t, :]

            i_t = self.i[:, t, :]
            f_t = self.f[:, t, :]
            c_tilde_t = self.c_tilde[:, t, :]
            o_t = self.o[:, t, :]

            h_prev = self.h_states[:, t, :]
            x_t = self.inp[:, t, :]

            tanh_c = cp.tanh(c_t)

            do = dh * tanh_c
            dc = dh * o_t * (1.0 - tanh_c ** 2) + dc_next

            di = dc * c_tilde_t
            df = dc * c_prev
            dc_tilde = dc * i_t
            dc_next = dc * f_t

            di_input = di * i_t * (1.0 - i_t)  # sigmoid
            df_input = df * f_t * (1.0 - f_t)
            do_input = do * o_t * (1.0 - o_t)
            dc_tilde_input = dc_tilde * (1.0 - c_tilde_t ** 2)  # tanh


            self.grad_W_i += x_t.T @ di_input
            self.grad_W_f += x_t.T @ df_input
            self.grad_W_c += x_t.T @ dc_tilde_input
            self.grad_W_o += x_t.T @ do_input

            self.grad_U_i += h_prev.T @ di_input
            self.grad_U_f += h_prev.T @ df_input
            self.grad_U_c += h_prev.T @ dc_tilde_input
            self.grad_U_o += h_prev.T @ do_input

            self.grad_b_i += cp.sum(di_input, axis=0)
            self.grad_b_f += cp.sum(df_input, axis=0)
            self.grad_b_c += cp.sum(dc_tilde_input, axis=0)
            self.grad_b_o += cp.sum(do_input, axis=0)

            grad_x[:, t, :] = (
                di_input @ self.W_i.T +
                df_input @ self.W_f.T +
                dc_tilde_input @ self.W_c.T +
                do_input @ self.W_o.T
            )

            dh_prev = (
                di_input @ self.U_i.T +
                df_input @ self.U_f.T +
                dc_tilde_input @ self.U_c.T +
                do_input @ self.U_o.T
            )

            dh_next = dh_prev

        return grad_x


    def update(self, lr: float, l2: float):
        self.W_i -= lr * (self.grad_W_i + l2 * self.W_i)
        self.W_f -= lr * (self.grad_W_f + l2 * self.W_f)
        self.W_c -= lr * (self.grad_W_c + l2 * self.W_c)
        self.W_o -= lr * (self.grad_W_o + l2 * self.W_o)

        self.U_i -= lr * (self.grad_U_i + l2 * self.U_i)
        self.U_f -= lr * (self.grad_U_f + l2 * self.U_f)
        self.U_c -= lr * (self.grad_U_c + l2 * self.U_c)
        self.U_o -= lr * (self.grad_U_o + l2 * self.U_o)

        self.b_i -= lr * self.grad_b_i
        self.b_f -= lr * self.grad_b_f
        self.b_c -= lr * self.grad_b_c
        self.b_o -= lr * self.grad_b_o
