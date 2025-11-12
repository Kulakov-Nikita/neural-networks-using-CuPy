import cupy as cp
from layers.abs_layer import Layer


class GRU(Layer):
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        limit = cp.sqrt(6 / (input_size + hidden_size))

        self.W_z = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_r = cp.random.uniform(-limit, limit, (input_size, hidden_size))
        self.W_h = cp.random.uniform(-limit, limit, (input_size, hidden_size))

        self.U_z = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_r = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.U_h = cp.random.uniform(-limit, limit, (hidden_size, hidden_size))

        self.b_z = cp.zeros(hidden_size, dtype=cp.float32)
        self.b_r = cp.zeros(hidden_size, dtype=cp.float32)
        self.b_h = cp.zeros(hidden_size, dtype=cp.float32)

        self.inp = None            
        self.h_states = None       
        self.z = None              
        self.r = None              
        self.h_tilde = None        

        self.grad_W_z = None
        self.grad_W_r = None
        self.grad_W_h = None

        self.grad_U_z = None
        self.grad_U_r = None
        self.grad_U_h = None

        self.grad_b_z = None
        self.grad_b_r = None
        self.grad_b_h = None


    def forward(self, x: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = x.shape
        self.inp = x
        dtype = x.dtype

        self.h_states = cp.zeros((batch_size, seq_len + 1, self.hidden_size), dtype=dtype)

        self.z = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
        self.r = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
        self.h_tilde = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)

        h_t = cp.zeros((batch_size, self.hidden_size), dtype=dtype)

        for t in range(seq_len):
            x_t = x[:, t, :]            # (batch_size, input_size)
            h_prev = h_t                # (batch_size, hidden_size)

            z_lin = x_t @ self.W_z + h_prev @ self.U_z + self.b_z
            z_t = 1.0 / (1.0 + cp.exp(-z_lin))

            r_lin = x_t @ self.W_r + h_prev @ self.U_r + self.b_r
            r_t = 1.0 / (1.0 + cp.exp(-r_lin)) 

            h_tilde_lin = x_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h
            h_tilde_t = cp.tanh(h_tilde_lin)

            h_t = (1.0 - z_t) * h_tilde_t + z_t * h_prev

            self.z[:, t, :] = z_t
            self.r[:, t, :] = r_t
            self.h_tilde[:, t, :] = h_tilde_t
            self.h_states[:, t + 1, :] = h_t

        outputs = self.h_states[:, 1:, :]  
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]          



    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        batch_size, seq_len, _ = self.inp.shape
        dtype = self.inp.dtype

        grad_x = cp.zeros_like(self.inp)

        self.grad_W_z = cp.zeros_like(self.W_z)
        self.grad_W_r = cp.zeros_like(self.W_r)
        self.grad_W_h = cp.zeros_like(self.W_h)

        self.grad_U_z = cp.zeros_like(self.U_z)
        self.grad_U_r = cp.zeros_like(self.U_r)
        self.grad_U_h = cp.zeros_like(self.U_h)

        self.grad_b_z = cp.zeros_like(self.b_z)
        self.grad_b_r = cp.zeros_like(self.b_r)
        self.grad_b_h = cp.zeros_like(self.b_h)

        if self.return_sequences:
            grad_h = grad_out
        else:
            grad_h = cp.zeros((batch_size, seq_len, self.hidden_size), dtype=dtype)
            grad_h[:, -1, :] = grad_out

        dh_next = cp.zeros((batch_size, self.hidden_size), dtype=dtype)

        for t in range(seq_len - 1, -1, -1):
            dh = grad_h[:, t, :] + dh_next 

            z_t = self.z[:, t, :]
            r_t = self.r[:, t, :]
            h_tilde_t = self.h_tilde[:, t, :]
            h_prev = self.h_states[:, t, :]
            x_t = self.inp[:, t, :]


            dh_tilde = dh * (1.0 - z_t)

            dz = dh * (h_prev - h_tilde_t)

            dh_prev = dh * z_t


            dh_tilde_input = dh_tilde * (1.0 - h_tilde_t ** 2)

            self.grad_W_h += x_t.T @ dh_tilde_input
            s = r_t * h_prev
            self.grad_U_h += s.T @ dh_tilde_input
            self.grad_b_h += cp.sum(dh_tilde_input, axis=0)

            ds = dh_tilde_input @ self.U_h.T      
            dr = ds * h_prev                      
            dh_prev += ds * r_t                   

            dr_input = dr * r_t * (1.0 - r_t)

            dz_input = dz * z_t * (1.0 - z_t)

            self.grad_W_r += x_t.T @ dr_input
            self.grad_W_z += x_t.T @ dz_input

            self.grad_U_r += h_prev.T @ dr_input
            self.grad_U_z += h_prev.T @ dz_input

            self.grad_b_r += cp.sum(dr_input, axis=0)
            self.grad_b_z += cp.sum(dz_input, axis=0)

            grad_x[:, t, :] = (
                dh_tilde_input @ self.W_h.T +
                dr_input @ self.W_r.T +
                dz_input @ self.W_z.T
            )

            dh_prev += dr_input @ self.U_r.T
            dh_prev += dz_input @ self.U_z.T

            dh_next = dh_prev

        return grad_x


    def update(self, lr: float, l2: float):
        self.W_z -= lr * (self.grad_W_z + l2 * self.W_z)
        self.W_r -= lr * (self.grad_W_r + l2 * self.W_r)
        self.W_h -= lr * (self.grad_W_h + l2 * self.W_h)

        self.U_z -= lr * (self.grad_U_z + l2 * self.U_z)
        self.U_r -= lr * (self.grad_U_r + l2 * self.U_r)
        self.U_h -= lr * (self.grad_U_h + l2 * self.U_h)

        self.b_z -= lr * self.grad_b_z
        self.b_r -= lr * self.grad_b_r
        self.b_h -= lr * self.grad_b_h
