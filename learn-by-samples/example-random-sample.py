"""
File: data.py
Author: Chuncheng Zhang
Date: 2023-05-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Train the diffusion model using randomized generated paired dataset.

Functions:
    1. Pending
    2. Pending
    3. Pending
    4. Pending
    5. Pending
"""


# %% ---- 2023-05-22 ------------------------
# Pending
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision


# %% ---- 2023-05-22 ------------------------
# Parameters

# Signal generation
T = 1  # Seconds
FS = 100  # Sample rate
NUM_FREQ = 5  # How many frequency
POINTS = int(T * FS)

# Diffusion
EPS = 1e-5
BETA_MAX = 0.1
DIFFUSION_STEPS = 100

# Training
NUM_EPOCHS = 5000
BATCH_SIZE = 2000
NUM_VALIDATE_ON_EPOCHS = 500

# %%
# Generate diffusion parameters
betas = torch.linspace(-6, 6, DIFFUSION_STEPS)
betas = torch.sigmoid(betas)*(BETA_MAX - EPS) + EPS

# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1-betas
alphas_prod = torch.cumprod(alphas, 0)
# 插入第一个数 1，丢掉最后一个数，previous连乘
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.numpy()
alphas_bar_sqrt = alphas_bar_sqrt.numpy()

plt.plot(one_minus_alphas_bar_sqrt, label='Noise factor, sqrt(1 - alpha_bar)')
plt.plot(alphas_bar_sqrt, label='Signal factor, sqrt(alpha_bar)')
plt.legend()
plt.grid(True)
plt.title('Diffusion trace for {} steps'.format(len(betas)))
plt.show()


# %%


def generate_signal(display_flag=False):
    # A reasonable frequency range
    available_freq = sorted(FS / 10 * np.random.random(NUM_FREQ))
    phis = np.random.random(NUM_FREQ) * np.pi

    times = np.linspace(0, T, POINTS)

    values = np.array([np.cos(times * freq * np.pi * 2 + phi)
                       for freq, phi in zip(available_freq, phis)])
    signal = np.sum(values, axis=0)
    signal /= np.std(signal)

    if display_flag:
        for v in values:
            plt.plot(times, v, color='#aaa5')

        plt.plot(times, signal)
        plt.title('Signal')
        plt.show()

    return signal, times


signals = []
for j in range(0):
    signal, t = generate_signal(display_flag=False)
    signals.append(signal)

signal, times = generate_signal(display_flag=True)
signals.append(signal)

# %% ---- 2023-05-22 ------------------------
# Pending


def diffuse(src, batch_size, dst=None):
    if dst is None:
        dst = np.random.randn(src.shape[0])

    data_t = []
    data_t_minus_1 = []

    steps = np.random.randint(1, DIFFUSION_STEPS, (batch_size,))

    for step in steps:
        r1 = alphas_bar_sqrt[step]
        r2 = one_minus_alphas_bar_sqrt[step]
        s = r1 * src + r2 * dst
        data_t.append(s)

        r1 = alphas_bar_sqrt[step-1]
        r2 = one_minus_alphas_bar_sqrt[step-1]
        s = r1 * src + r2 * dst
        data_t_minus_1.append(s)

    data_t = np.array(data_t)
    data_t_minus_1 = np.array(data_t_minus_1)

    return data_t, data_t_minus_1, steps


data_t, data_t_minus_1, steps = diffuse(signal, batch_size=200)
data_t.shape, data_t_minus_1.shape, steps.shape


# %% ---- 2023-05-22 ------------------------
# Define the network


class WeightClipper(object):
    def __init__(self):
        pass

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w


class Net(nn.Module):
    def __init__(self, dimension=POINTS):
        super().__init__()

        core_size = 100

        self.mlp1 = torchvision.ops.MLP(
            dimension, [100, core_size], activation_layer=nn.LeakyReLU)

        self.mlp2 = torchvision.ops.MLP(
            core_size, [100, dimension], activation_layer=nn.LeakyReLU)

        self.embedding = nn.Embedding(DIFFUSION_STEPS, core_size)

    def forward(self, x, step):
        x = self.mlp1(x)
        x += self.embedding(step)
        x = self.mlp2(x)
        return x


net = Net().cuda()
clipper = WeightClipper()
net.apply(clipper)

net

# %%
# Validation


def validation(sup_title='Sup title', loss_trace=None):
    '''
    Validate the net
    '''
    n = 6
    noise = np.random.randn(n, POINTS)
    noise = torch.Tensor(noise).cuda()
    print(noise.shape)

    with torch.no_grad():
        for step in range(DIFFUSION_STEPS):
            s = torch.zeros(n).to(torch.int32) + step
            noise = net(noise, s.cuda())

        data = noise.cpu().numpy()
        print(data.shape)

    # Curves
    fig, axs = plt.subplots(n//2 + 1, 2, figsize=(10, 15))
    axs = axs.ravel()

    for j, ax in zip(range(n), axs[:n]):
        for s in signals:
            ax.plot(times, s, linewidth=0.5)
        ax.plot(times, data[j], linewidth=2, color='#f005')
        ax.set_title(f'Diffused: {j}')

    # Loss trace
    if loss_trace is not None:
        ax = axs[n]
        ax.plot(loss_trace, label='Loss')
        ax.legend()
        ax.set_title('Loss')

    fig.suptitle(sup_title)
    fig.tight_layout()
    plt.show()


validation('Not trained')


# %%
# Train the network

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

loss_trace = []
for epoch in range(NUM_EPOCHS + 1):
    data_t, data_t_minus_1, steps = diffuse(signal, batch_size=BATCH_SIZE)

    X = torch.Tensor(data_t).cuda()
    y = torch.Tensor(data_t_minus_1).cuda()
    steps = torch.Tensor(steps).to(torch.int32).cuda()

    p = net(X, steps)

    loss = criterion(p, y)
    v = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_trace.append(v)
    if epoch % 50 == 0:
        print(epoch, v)

    if epoch > 0 and epoch % NUM_VALIDATE_ON_EPOCHS == 0:
        validation(f'Epoch: {epoch}', loss_trace)


# %%
validation('Final', loss_trace)

# %%
