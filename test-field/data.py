"""
File: data.py
Author: Chuncheng Zhang
Date: 2023-05-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Generation of the diffusion simulation dataset.

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
DIFFUSION_REPEAT_TIMES = 2000

# Training
NUM_EPOCHS = 4000
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


class Scaler(object):
    def __init__(self, k, b):
        self.k = k
        self.k_inv = 1 / k
        self.b = b

    def transform(self, x):
        return x * self.k + self.b

    def inv_transform(self, x):
        return (x - self.b) * self.k_inv


SCALER = Scaler(k=1 / (NUM_FREQ*2), b=0.5)

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


def diffuse(src, dst=None, steps=DIFFUSION_STEPS, debug_flag=False):
    '''
    Diffuse from src to dst

    Args:
        src: numpy array: The diffusion process starts from src to dst.
        dst: numpy array: The diffusion process starts from src to dst, if it is None, use random.
        beta_max: float: The max of the beta parameter for the diffusion process, since it becomes faster and faster during the diffusing process.
        steps: int: How many steps to diffuse.
        debug_flag: boolean: Whether returns everything as debug.

    Return:
        diffusion: numpy array: The diffusion process in the matrix of [steps x time points]
    '''
    if dst is None:
        dst = np.random.randn(src.shape[0])

    diffusion = []
    distance = []

    for step in range(steps):
        r1 = alphas_bar_sqrt[step]
        r2 = one_minus_alphas_bar_sqrt[step]
        s = r1 * src + r2 * dst
        diffusion.append(s)

        if debug_flag:
            distance.append(np.linalg.norm(s - src))

    diffusion = np.array(diffusion[::-1])

    if debug_flag:
        distance = np.array(distance[::-1])

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        gs = axs[0, 0].get_gridspec()
        [ax.remove() for ax in axs[0, 0:]]

        ax = fig.add_subplot(gs[0, 0:])
        d = diffusion[::-int(diffusion.shape[0]/10)]
        for i in range(1, len(d)):
            ax.plot(times, d[i])
        ax.plot(times, diffusion[-1] + 0.01, linewidth=2)
        ax.plot(times, d[0], linewidth=2)
        ax.set_title('Diffusion (forward process)')

        ax = axs[1, 0]
        ax.hist(diffusion[-1], color='#f008', label='src')
        ax.hist(diffusion[0], color='#0f08', label='dst')
        ax.legend(loc='best')
        ax.set_title('Diffusion histogram')

        ax = axs[1, 1]
        ax.plot(distance, label='Distance', color='darkgrey')
        ax2 = ax.twinx()
        ax2.plot(alphas_bar_sqrt, label='alphas_bar_sqrt', color='red')
        ax2.plot(one_minus_alphas_bar_sqrt,
                 label='one_minus_alphas_bar_sqrt', color='blue')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax2.legend()
        ax.set_title('Diffusion trace')

        fig.tight_layout()
        plt.show()

    return diffusion


diffusion = diffuse(
    signal, dst=None, steps=DIFFUSION_STEPS, debug_flag=True)


# %% ---- 2023-05-22 ------------------------
# Pending


# %%
# Generate the training data

'''
The train_data has the dimension of (n x POINTS),
it is the (DIFFUSION_STEPS x POINTS) matrix repeats DIFFUSION_REPEAT_TIMES times.
'''

train_data = np.concatenate([SCALER.transform(diffuse(random.choice(signals), steps=DIFFUSION_STEPS))
                            for _ in range(DIFFUSION_REPEAT_TIMES)], axis=0)

train_diffusion_step = [j % DIFFUSION_STEPS
                        for j in range(train_data.shape[0])]

index_vector = [j for j in range(train_data.shape[0])
                if j % DIFFUSION_STEPS != (DIFFUSION_STEPS-1)]

# %%
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


# class Net(nn.Module):
#     def __init__(self, dimension=POINTS):
#         super().__init__()
#         self.mlp = torchvision.ops.MLP(
#             dimension, [100, 50, 100, dimension], activation_layer=nn.LeakyReLU)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         output = self.mlp(x)
#         output = self.act(output)
#         return output


class Net(nn.Module):
    def __init__(self, dimension=POINTS):
        super().__init__()

        core_size = 50

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
    noise = SCALER.transform(np.random.randn(n, POINTS))
    noise = torch.Tensor(noise).cuda()
    print(noise.shape)

    with torch.no_grad():
        for step in range(DIFFUSION_STEPS):
            s = torch.zeros(n).to(torch.int32) + step
            noise = net(noise, s.cuda())

        data = SCALER.inv_transform(noise.cpu().numpy())
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
        ax.plot([e[0] for e in loss_trace], label='Loss sum')
        ax.plot([e[1] for e in loss_trace], label='Loss 1')
        ax.plot([e[2] for e in loss_trace], label='Loss 2')
        ax.legend()
        ax.set_title('Loss')

    fig.suptitle(sup_title)
    fig.tight_layout()
    plt.show()


validation('Not trained')


# %%
# Train the network

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
criterion2 = nn.KLDivLoss(reduction='batchmean')
# criterion = nn.HuberLoss()

# Training data
data_X = torch.Tensor(train_data[:-1]).cuda()
data_y = torch.Tensor(train_data[1:]).cuda()
data_step = torch.Tensor(train_diffusion_step).to(torch.int32).cuda()

loss_trace = []
for epoch in range(NUM_EPOCHS + 1):
    np.random.shuffle(index_vector)

    select = index_vector[:BATCH_SIZE]
    X = data_X[select]
    y = data_y[select]
    step = data_step[select]
    p = net(X, step)

    loss1 = criterion(p, y)
    loss2 = criterion2(p.log_softmax(dim=-1), y.softmax(dim=-1))
    loss = loss1 + 0.0 * loss2
    v1 = loss1.item()
    v2 = loss2.item()
    v = loss.item()

    optimizer.zero_grad()
    loss.backward()
    # net.apply(clipper)
    optimizer.step()

    loss_trace.append([v, v1, v2])
    if epoch % 50 == 0:
        print(epoch, v, v1, v2)

    if epoch > 0 and epoch % NUM_VALIDATE_ON_EPOCHS == 0:
        validation(f'Epoch: {epoch}', loss_trace)


# %%
validation('Final', loss_trace)

# %%
