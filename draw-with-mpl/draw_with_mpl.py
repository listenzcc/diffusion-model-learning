
"""
File: draw_with_mpl.py
Author: Chuncheng Zhang
Date: 2023-05-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Learn how to draw with matplotlib(mpl)

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-05-24 ------------------------
# Requirements and constants

# Signal generation
# Best parameters
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Constants
T = 1  # Seconds
FS = 100  # Sample rate
NUM_FREQ = 5  # How many frequency
POINTS = int(T * FS)
RANDOM_POPULATION = 20


# %% ---- 2023-05-24 ------------------------
# Function and class

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

    # Plot, if required
    if display_flag:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for j, v in enumerate(components):
            ax.plot(times, v, color='#aaa5',
                    label='Components' if j == 0 else None)

        ax.plot(times, signal, label='Signal')
        ax.set_title('Signal')
        ax.legend()
        fig.tight_layout()
        plt.show()

    return signal


# %% ---- 2023-05-24 ------------------------
# Play ground
signal = generate_signal(display_flag=True)
signal


# %% ---- 2023-05-24 ------------------------
# Pending
cmap = plt.cm.Blues
cmap

# %%


def draw_signal_with_I_cov(signal: np.array, sigma: float, ax: plt.Axes, use_mu_flag=True, legend_flag=True):
    """
    Draw the signal with I covariance matrix.

    Args:
        signal (np.Array): The signal to draw in red line, and the noise are added by sigma.
        sigma (float): The strength of the variance.
        ax (plt.Axes): The axes to draw the signal with.
        use_mu_flag (bool): Whether to draw the signal's noise by its mu. Default is True.
        legend_flag (bool): Whether to draw the legend. Defaults to True.

    Returns:
        data: The generated signal plus noise.
    """

    mu = np.repeat(signal[np.newaxis, :], RANDOM_POPULATION, axis=0)

    if not use_mu_flag:
        mu *= 0

    data = mu + np.random.randn(RANDOM_POPULATION, POINTS) * sigma
    distance = [np.linalg.norm(d - signal) for d in data]

    def sort_by(iterable, by, reverse=False):
        by = np.array(by)
        ind = np.argsort(by)
        if reverse:
            ind = ind[::-1]
        return iterable[ind], by[ind]

    def opacity(color, opacity=0.5):
        color = list(color)
        if len(color) == 3:
            color.append(opacity)
        else:
            color[3] = opacity
        return tuple(color)

    data, distance = sort_by(data, distance, reverse=True)

    norm = mpl.colors.Normalize(vmin=np.min(distance), vmax=np.max(distance))
    for y, d in zip(data, distance):
        ax.plot(y, color=opacity(cmap(norm(d)), 0.5))
    # The latest drawn line is on the most top
    ax.plot(signal, color='red', label='Signal')

    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    if legend_flag:
        ax.legend(loc='lower right')

    ax.set_facecolor('gray')
    ax.set_title('Signal and its noise: (sigma={})'.format(sigma))

    return data


# Draw the signal with its mu
fig, axs = plt.subplots(3, 1, figsize=(6, 6))
draw_signal_with_I_cov(signal, 1.0, axs[0], legend_flag=False)
draw_signal_with_I_cov(signal, 0.5, axs[1], legend_flag=False)
draw_signal_with_I_cov(signal, 0.1, axs[2], legend_flag=True)
fig.tight_layout()
plt.show()

# Draw the signal without its mu
fig, axs = plt.subplots(3, 1, figsize=(6, 3))
draw_signal_with_I_cov(signal, 0.1, axs[2], use_mu_flag=False)
fig.tight_layout()
plt.show()

# %%
im = plt.imshow(signal.reshape(10, 10))
plt.title('Signal in rect')
plt.colorbar(im)

# %%
