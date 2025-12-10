import cupy as cp

from layers.dense import Dense
from layers.activations import ReLU


def sigmoid(x: cp.ndarray) -> cp.ndarray:
    return 1 / (1 + cp.exp(-x))


def bce_logits_loss(logits: cp.ndarray, targets: cp.ndarray):
    preds = sigmoid(logits)
    eps = 1e-7
    loss = -cp.mean(targets * cp.log(preds + eps) + (1 - targets) * cp.log(1 - preds + eps))
    grad = (preds - targets) / len(logits)
    return loss, grad


def reconstruction_bce(recon: cp.ndarray, x: cp.ndarray):
    eps = 1e-7
    loss = -cp.mean(x * cp.log(recon + eps) + (1 - x) * cp.log(1 - recon + eps))
    grad = (recon - x) / recon.size
    return loss, grad


def kl_divergence(mean: cp.ndarray, logvar: cp.ndarray):
    kl = -0.5 * cp.mean(1 + logvar - mean ** 2 - cp.exp(logvar))
    d_mean = mean / mean.size
    d_logvar = 0.5 * (cp.exp(logvar) - 1) / logvar.size
    return kl, d_mean, d_logvar


def iterate_minibatches(X, y, batch_size, shuffle=True):
    n = len(X)
    indices = cp.arange(n)
    if shuffle:
        cp.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def update_layers(layers, lr, l2=0.0):
    for layer in layers:
        layer.update(lr, l2)


def concat_label(x, y, conditional, num_classes=10):
    if not conditional:
        return x
    y_oh = cp.zeros((len(y), num_classes), dtype=cp.float32)
    y_oh[cp.arange(len(y)), y.astype(cp.int32)] = 1.0
    return cp.concatenate([x, y_oh], axis=1)


class Discriminator:
    def __init__(self, input_dim: int, conditional: bool = False, num_classes: int = 10):
        self.input_dim = input_dim
        self.conditional = conditional
        self.num_classes = num_classes
        extra = num_classes if conditional else 0
        self.layers = [
            Dense(input_dim + extra, 256),
            ReLU(),
            Dense(256, 128),
            ReLU(),
        ]
        self.out = Dense(128, 1)

    def forward(self, x: cp.ndarray, y: cp.ndarray | None):
        x_in = concat_label(x, y, self.conditional, self.num_classes)
        h = x_in
        for layer in self.layers:
            h = layer.forward(h)
        logits = self.out.forward(h).squeeze()
        return logits

    def backward(self, grad_logits: cp.ndarray):
        grad = grad_logits.reshape(-1, 1)
        grad_h = self.out.backward(grad)
        for layer in reversed(self.layers):
            grad_h = layer.backward(grad_h)
        grad_x = grad_h[:, : self.input_dim] if self.conditional else grad_h
        return grad_x

    def update(self, lr: float, l2: float = 0.0):
        update_layers(self.layers + [self.out], lr, l2)


class VAE:
    def __init__(self, input_dim: int, latent_dim: int, conditional: bool = False, num_classes: int = 10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.num_classes = num_classes
        extra = num_classes if conditional else 0

        self.enc_base = [Dense(input_dim + extra, 256), ReLU(), Dense(256, 128), ReLU()]
        self.enc_mean = Dense(128, latent_dim)
        self.enc_logvar = Dense(128, latent_dim)

        self.dec_layers = [Dense(latent_dim + extra, 128), ReLU(), Dense(128, 256), ReLU()]
        self.dec_out = Dense(256, input_dim)

        self.std = None
        self.eps = None
        self.recon = None

    def forward(self, x: cp.ndarray, y: cp.ndarray | None):
        x_in = concat_label(x, y, self.conditional, self.num_classes)
        h = x_in
        for layer in self.enc_base:
            h = layer.forward(h)
        mean = self.enc_mean.forward(h)
        logvar = self.enc_logvar.forward(h)

        self.eps = cp.random.randn(*mean.shape).astype(cp.float32)
        self.std = cp.exp(0.5 * logvar)
        z = mean + self.std * self.eps

        z_in = concat_label(z, y, self.conditional, self.num_classes)
        h_dec = z_in
        for layer in self.dec_layers:
            h_dec = layer.forward(h_dec)
        logits = self.dec_out.forward(h_dec)
        self.recon = sigmoid(logits)
        return self.recon, mean, logvar

    def backward(self, grad_recon: cp.ndarray, d_mean: cp.ndarray, d_logvar: cp.ndarray):
        grad_logits = grad_recon * self.recon * (1 - self.recon)
        grad_h = self.dec_out.backward(grad_logits)
        for layer in reversed(self.dec_layers):
            grad_h = layer.backward(grad_h)

        grad_z_in = grad_h
        grad_z = grad_z_in[:, : self.latent_dim] if self.conditional else grad_z_in

        d_mean = d_mean + grad_z
        d_logvar = d_logvar + grad_z * (0.5 * self.std * self.eps)

        grad_enc = self.enc_mean.backward(d_mean) + self.enc_logvar.backward(d_logvar)
        for layer in reversed(self.enc_base):
            grad_enc = layer.backward(grad_enc)

    def update(self, lr: float, l2: float = 0.0):
        update_layers(self.dec_layers + [self.dec_out], lr, l2)
        update_layers(self.enc_base + [self.enc_mean, self.enc_logvar], lr, l2)

    def sample(self, num_samples: int = 16, labels: cp.ndarray | None = None):
        z = cp.random.randn(num_samples, self.latent_dim).astype(cp.float32)
        z_in = concat_label(z, labels, self.conditional, self.num_classes)
        h = z_in
        for layer in self.dec_layers:
            h = layer.forward(h)
        logits = self.dec_out.forward(h)
        recon = sigmoid(logits)
        return recon.reshape(-1, 1, 28, 28)


def train_vae(train_X, train_y, test_X, test_y, latent_dim=16, epochs=6, batch_size=256, lr=1e-3, conditional=False, l2=0.0):
    input_dim = train_X.shape[1]
    vae = VAE(input_dim, latent_dim, conditional=conditional)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for xb, yb in iterate_minibatches(train_X, train_y, batch_size, shuffle=True):
            recon, mean, logvar = vae.forward(xb, yb if conditional else None)
            rec_loss, grad_recon = reconstruction_bce(recon, xb)
            kl, d_mean, d_logvar = kl_divergence(mean, logvar)

            vae.backward(grad_recon, d_mean, d_logvar)
            vae.update(lr, l2)

        for xb, yb in iterate_minibatches(test_X, test_y, batch_size, shuffle=False):
            recon, mean, logvar = vae.forward(xb, yb if conditional else None)
            rec_loss, grad_recon = reconstruction_bce(recon, xb)
            kl, d_mean, d_logvar = kl_divergence(mean, logvar)

            loss = rec_loss + kl
            epoch_loss += loss * len(xb)

        losses.append(epoch_loss / len(test_X))
        print(f"VAE epoch {epoch + 1}/{epochs}: loss={losses[-1]:.4f}")
    return vae, losses


def train_vae_gan(
    data_X,
    data_y,
    latent_dim=16,
    epochs=6,
    batch_size=256,
    lr_vae=1e-3,
    lr_d=5e-4,
    lambda_gan=0.5,
    conditional=False,
    l2=0.0,
):
    input_dim = data_X.shape[1]
    vae = VAE(input_dim, latent_dim, conditional=conditional)
    disc = Discriminator(input_dim, conditional=conditional)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in iterate_minibatches(data_X, data_y, batch_size, shuffle=True):
            recon, mean, logvar = vae.forward(xb, yb if conditional else None)

            # Discriminator step
            disc_input = cp.concatenate([xb, recon], axis=0)
            disc_labels = cp.concatenate([cp.ones(len(xb), dtype=cp.float32), cp.zeros(len(xb), dtype=cp.float32)], axis=0)
            disc_y = cp.concatenate([yb, yb], axis=0) if conditional else None
            disc_logits = disc.forward(disc_input, disc_y)
            d_loss, d_grad = bce_logits_loss(disc_logits, disc_labels)
            disc.backward(d_grad)
            disc.update(lr_d, l2)

            # VAE / generator step
            rec_loss, grad_recon = reconstruction_bce(recon, xb)
            kl, d_mean, d_logvar = kl_divergence(mean, logvar)

            fake_logits = disc.forward(recon, yb if conditional else None)
            adv_loss, adv_grad_logits = bce_logits_loss(fake_logits, cp.ones_like(fake_logits))
            grad_adv = disc.backward(adv_grad_logits)

            grad_recon_total = grad_recon + lambda_gan * grad_adv
            vae.backward(grad_recon_total, d_mean, d_logvar)
            vae.update(lr_vae, l2)

            total_loss = rec_loss + kl + lambda_gan * adv_loss
            epoch_loss += total_loss * len(xb)

        losses.append(epoch_loss / len(data_X))
        print(f"VAE+GAN epoch {epoch + 1}/{epochs}: loss={losses[-1]:.4f}")
    return vae, disc, losses
