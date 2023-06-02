"""
File: practise.py
Author: Chuncheng Zhang
Date: 2023-05-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The aim is to generate the signal using diffusion model

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-05-24 ------------------------
# Imports and constants

import torch
import torch.nn as nn
import torchvision

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Signal generation
# Best parameters
T = 1  # Seconds
FS = 50  # Sample rate
NUM_FREQ = 5  # How many frequency
POINTS = int(T * FS)

# Diffusion
EPS = 1e-5
BETA_MAX = 0.1
DIFFUSION_STEPS = 100


# %% ---- 2023-05-24 ------------------------
# Functions
def generate_signal(display_flag=False):
    """
    Generate random signal

    Args:
        display_flag (bool, optional): Whether plot the signal. Defaults to False.

    Returns:
        signal: The generated signal.
    """

    # Random frequency and phase
    available_freq = sorted(FS / 10 * np.random.random(NUM_FREQ))
    phis = np.random.random(NUM_FREQ) * np.pi

    # Time line
    times = np.linspace(0, T, POINTS)

    # Components and signal
    components = np.array([np.cos(times * freq * np.pi * 2 + phi)
                           for freq, phi in zip(available_freq, phis)])
    signal = np.sum(components, axis=0)
    signal /= np.std(signal)

    # Convert signal to tensor
    signal = torch.Tensor(signal)

    # Plot, if required
    if display_flag:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for v in components:
            ax.plot(times, v, color='#aaa5')

        ax.plot(times, signal)
        ax.set_title('Signal')
        fig.tight_layout()
        plt.show()

    return signal


def plot_diffusion(signal: torch.Tensor, steps: torch.Tensor, noise: torch.Tensor, result: torch.Tensor):
    """
    Plot the diffusion results,
    the diffusion process is the signal diffuses by steps to noise,
    and the result is the diffusion results.

    Args:
        signal (torch.Tensor): The signal.
        steps (torch.Tensor): The steps.
        noise (torch.Tensor): The noise.
        result (torch.Tensor): The result.
    """
    cmap = plt.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=DIFFUSION_STEPS)

    sort = steps.sort()
    ind = sort.indices[-1]
    step = steps[ind]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    ax = axs[0]
    for j in range(batch_size):
        ax.plot(result[:, j], color=cmap(norm(steps[j])))
    ax.plot(result[:, ind], color='darkred', label=f'Farest {step}')
    ax.plot(noise, color='black', label='Noise')

    color = cmap(norm(0))
    color = [e for e in color]
    color[-1] = 0.2
    ax.plot(signal, color=color, linewidth=10, label='Signal')
    ax.legend()
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    ax.set_title('Diffusion results, Farest and Noise should be of similar')

    ax = axs[1]
    ax.hist(signal, color=cmap(norm(0)), rwidth=0.5, label='Signal')
    ax.hist(result[:, ind], color='#a338', rwidth=0.5, label=f'Farest {step}')
    ax.legend()
    ax.set_title('Histogram')

    fig.tight_layout()

    plt.show()

    return fig


def diffuse_to(signal: torch.Tensor,
               steps: torch.Tensor,
               alphas_bar_sqrt: torch.Tensor,
               one_minus_alphas_bar_sqrt: torch.Tensor,
               debug_flag=False):
    """
    Diffuse the signal with given steps,
    and the noise is randomly generated with the same shape of the signal.

    ---
    The signal is the (POINTS, ) tensor refers the POINTS length signal.
    The noise has the same shape as the signal.
    The steps is the (n, ) tensor refers the steps of interest.
    The result i the (POINTS, n) tensor refers the diffused results.

    ---
    The n refers the batch size,
    and signal and noise are stacked to generate (POINTS, n) tensor by repeating them.

    result = signal * r1 + noise * r2

    ---
    The ratio r1 and r2 are derived from alphas_bar_sqrt and one_minus_alphas_bar_sqrt accordingly.

    Args:
        signal (torch.Tensor): The raw signal to diffuse.
        steps (torch.Tensor): The steps.
        alphas_bar_sqrt (torch.Tensor): The factor of the signal.
        one_minus_alphas_bar_sqrt (torch.Tensor): The factor of the noise.
        plot_flag (bool): Whether to plot the result for debug. Default to False.

    Returns:
        result: The diffusion outcome.
        noise: The noise.
        batch_noise: The batched noise.
    """

    # The steps' length determines the batch_size
    batch_size = steps.shape[0]

    def mk_batch(signal: torch.Tensor, batch_size=batch_size):
        batch = torch.repeat_interleave(
            signal.reshape(len(signal), 1), batch_size, axis=1)
        return batch

    batch_signal = mk_batch(signal)
    noise = signal * 0
    batch_noise = torch.randn_like(batch_signal)

    if debug_flag:
        noise = torch.randn_like(signal)
        batch_noise = mk_batch(noise)

    r1 = alphas_bar_sqrt[steps]
    r2 = one_minus_alphas_bar_sqrt[steps]

    result = batch_signal * r1 + batch_noise * r2

    if debug_flag:
        plot_diffusion(signal, steps, noise, result)

    return result, noise, batch_noise


def generate_diffusion_parameters(display_flag=True):
    """
    Generate the diffusion parameters for a given

    Args:
        display_flag (bool, optional): Whether to plot the parameters' changing along the diffusion steps. Defaults to True.

    Returns:
        betas: The tensor array of the diffusion parameters
        alphas_bar_sqrt: The tensor array of the diffusion parameters
        one_minus_alphas_bar_sqrt: The tensor array of the diffusion parameters
    """

    # Generate diffusion parameters
    betas = torch.linspace(-6, 6, DIFFUSION_STEPS)
    betas = torch.sigmoid(betas)*(BETA_MAX - EPS) + EPS

    # Generate alpha, alpha_prod, alpha_prod_previous, alpha_bar_sqrt, ...
    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas, 0)

    # Link the alphas_prod with [1]
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    if display_flag:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(one_minus_alphas_bar_sqrt,
                label='Noise factor, one_minus_alphas_bar_sqrt')
        ax.plot(alphas_bar_sqrt, label='Signal factor, alphas_bar_sqrt')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Diffusion step')
        ax.set_title('Diffusion trace for {} steps'.format(len(betas)))
        fig.tight_layout()
        plt.show()

    return betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt


def p_sample_loop(net, betas, one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""

    betas = betas.cuda()
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.cuda()

    cur_x = torch.randn((1, POINTS)).cuda()
    x_seq = [cur_x]
    for i in reversed(range(DIFFUSION_STEPS)):
        cur_x = p_sample(net, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从 x_t 开始生成 t-1 时刻的重构值"""
    t = torch.tensor([t]).cuda()

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x).cuda()
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    return (sample)


class Net(nn.Module):
    """
    Define the network
    """

    def __init__(self, dimension=POINTS):
        super().__init__()

        layer1_size = dimension * 2
        core_size = layer1_size * 2

        self.mlp1 = torchvision.ops.MLP(
            dimension, [layer1_size, core_size], activation_layer=nn.LeakyReLU)

        self.mlp2 = torchvision.ops.MLP(
            core_size, [layer1_size, dimension], activation_layer=nn.LeakyReLU)

        self.embedding = nn.Embedding(DIFFUSION_STEPS, core_size)

    def forward(self, x, step):
        x = self.mlp1(x)
        x += self.embedding(step)
        x = self.mlp2(x)
        return x


# %% ---- 2023-05-24 ------------------------
# Pending
signal = generate_signal(display_flag=True)

betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = generate_diffusion_parameters()


# %% ---- 2023-05-24 ------------------------
# Explain diffuse
batch_size = 10

steps = torch.randint(0, DIFFUSION_STEPS, (batch_size,))

result, noise, batch_noise = diffuse_to(signal,
                                        steps,
                                        alphas_bar_sqrt,
                                        one_minus_alphas_bar_sqrt,
                                        debug_flag=True)

print(result.shape, noise.shape, batch_noise.shape)

# %%
net = Net().cuda()
net

# %%
# optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# %%
NUM_EPOCHS = 5000
BATCH_SIZE = 1000

loss_trace = []

# for round in range(10):
for round in range(3):
    for epoch in range(NUM_EPOCHS):

        steps = torch.randint(0, DIFFUSION_STEPS, (batch_size,))

        result, noise, batch_noise = diffuse_to(
            signal,
            steps,
            alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt)

        X = result.T.cuda()
        y = batch_noise.T.cuda()
        s = steps.cuda()

        p = net(X, s)

        loss = criterion(p, y)
        v = loss.item()
        loss_trace.append(v)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()

        if epoch % 50 == 0:
            print(epoch, v)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    ax = axs[0]
    ax.plot(loss_trace)
    ax.set_title(f'Loss:')

    with torch.no_grad():
        seq = p_sample_loop(net, betas, one_minus_alphas_bar_sqrt)
        output = np.array([e.cpu().numpy() for e in seq]).squeeze()

    ax = axs[1]
    ax.plot(output)

    ax = axs[2]
    ax.plot(output[0], label='Noise: 0')
    ax.plot(output[-1], label='Noise: -1')
    ax.plot(signal, label='Signal')
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'Round-{round}.png')
    plt.show()


# %%

with torch.no_grad():
    seq = p_sample_loop(net, betas, one_minus_alphas_bar_sqrt)
    output = np.array([e.cpu().numpy() for e in seq]).squeeze()

print(output.shape)


cmap = plt.cm.Blues
norm = mpl.colors.Normalize(vmin=0, vmax=DIFFUSION_STEPS)

fig, axs = plt.subplots(2, 1, figsize=(6, 6))

ax = axs[0]
for step, vec in enumerate(output):
    ax.plot(vec, color=cmap(norm(step)))
ax.plot(signal, color='red', linewidth=0.5)
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
ax.set_xlabel('Dimension')
ax.set_title('Curves')

ax = axs[1]
ax.plot(output)
ax.set_xlabel('Diffusion step')
ax.set_title('Dimension change by steps')

fig.tight_layout()
fig.savefig('final.png')
plt.show()


# %%
