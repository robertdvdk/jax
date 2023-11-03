"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import numpy as np
import optax
import torchvision.datasets
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.distributions.multivariate_normal import MultivariateNormal

# Function definitions
class VAE(nn.Module):
    input_size : int
    hidden_size : int
    latent_size : int

    def setup(self):
        self.linear1 = nn.Dense(self.hidden_size)
        self.mean = nn.Dense(self.latent_size)
        self.logvar_layer = nn.Dense(self.latent_size)

        self.linear2 = nn.Dense(self.hidden_size)
        self.out = nn.Dense(self.input_size)


    def __call__(self, x, znoise, take_sample=False):
        if not take_sample:
            x = nn.relu(self.linear1(x))
            zmean = self.mean(x)
            zstd = jnp.sqrt(jnp.exp(self.logvar_layer(x)))

            z = zmean + zstd * znoise

            y = nn.relu(self.linear2(z))
            y = nn.sigmoid(self.out(y))
            return y, zmean, zstd
        else:
            y = nn.relu(self.linear2(znoise))
            y = nn.sigmoid(self.out(y))
            return y

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def calculate_loss(state, params, data_input, znoise):
    data_input = jnp.reshape(data_input, (data_input.shape[0], -1))

    y, zmean, zstd = state.apply_fn(params, data_input, znoise)
    L1 = -0.5 * jnp.sum(1 + jnp.log(jnp.square(zstd)) - jnp.square(zmean) - jnp.square(zstd))

    # We want the sum of squared errors, so we take 2 times the l2 loss.
    L2 = 2*optax.l2_loss(y, data_input).sum()
    return L1 + L2, (L1, L2)

@jax.jit
def train_step(state, batch, znoise):
    grad_fn = jax.value_and_grad(calculate_loss,
                                 argnums=1,
                                 has_aux=True)
    (L, (L1, L2)), grads = grad_fn(state, state.params, batch, znoise)
    state = state.apply_gradients(grads=grads)
    return state, L1, L2


import time
def train_model(state, data_loader, key, batch_size, num_epochs=100):
    for epoch in range(num_epochs):
        for x, t in data_loader:
            x = x.astype(np.float32) / 255
            key = jax.random.split(key, num=1)[0]
            znoise = jax.random.normal(key, (batch_size, 10))
            state, L1, L2 = train_step(state, x, znoise)

        sample = x[0]
        plt.imshow(sample)
        plt.savefig(f'./results/sample_{epoch}.png')
        y, _, _ = state.apply_fn(state.params, jnp.reshape(sample, 784), znoise[0])
        plt.imshow(jnp.reshape(y, (28, 28)))
        plt.savefig(f'./results/recon_{epoch}.png')
        gen = state.apply_fn(state.params, jnp.zeros(784), znoise[1], take_sample=True)
        plt.imshow(jnp.reshape(gen, (28, 28)))
        plt.savefig(f'./results/gen_{epoch}.png')

    return state


# Function definitions
def main():
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    batch_size = 256
    mnist = torchvision.datasets.MNIST('./data/', train=True, download=True)
    loader = torch.utils.data.DataLoader(mnist, batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
    net = VAE(784, 500, 10)
    rng = jax.random.PRNGKey(10)
    rng, inp_rng, init_rng, noise_rng, train_rng = jax.random.split(rng, 5)
    inp = jax.random.normal(inp_rng, (batch_size, 784))
    znoise = jax.random.normal(noise_rng, (batch_size, 10))
    params = net.init(init_rng, inp, znoise)
    schedule = optax.exponential_decay(0.001, 5 * (len(mnist) / batch_size), 0.9, staircase=True)
    optimizer = optax.adamw(learning_rate=schedule)

    model_state = train_state.TrainState.create(apply_fn=net.apply,
                                                params=params,
                                                tx=optimizer)

    train_model(model_state, loader, train_rng, batch_size)

if __name__ == "__main__":
    main()
