
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
RANDOM_POPULATION = 50


CMAP = plt.cm.Blues
# Use ipython's display function to display the cmap
try:
    display(CMAP)
except:
    pass

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


def sort_by(iterable, by, reverse=False):
    """
    Sort iterable with by vector,
    of course, they are of the same length.

    Args:
        iterable (iter): The iterable being sorted.
        by (iter): It is sorted firstly, and the iterable is sorted according to the order.
        reverse (bool, optional): Whether the sort is inverted. Defaults to False.

    Returns:
        sorted iterable, sorted by: The sorted results.
    """
    by = np.array(by)
    ind = np.argsort(by)
    if reverse:
        ind = ind[::-1]
    return iterable[ind], by[ind]


def opacity(color, opacity=0.5):
    """
    Add or set the opacity channel of the color,
    the color is 3x or 4x float.

    Args:
        color (3x or 4x array): The color in rgb or rgba.
        opacity (float, optional): The opacity value. Defaults to 0.5.

    Returns:
        The new color: The color being operated.
    """
    color = list(color)
    if len(color) == 3:
        color.append(opacity)
    else:
        color[3] = opacity
    return tuple(color)


def generate_noise_for_signal(signal: np.array,
                              sigma: float,
                              ax=None,  # : plt.Axes,
                              cmap=None,  # : mpl.colors.LinearSegmentedColormap,
                              dst_cov=None,
                              zero_mu_flag=False,
                              legend_flag=True):
    """
    Add the noise into signal with noise. 

    I covariance matrix.

    The signal is the 1-D array with POINTS samples.

    The data is ordered by their distance from tbe signal,
    the order is farthest first.

    Args:
        signal (np.Array): The signal to draw in red line, and the noise are added by sigma.
        sigma (float): The strength of the variance.
        ax (plt.Axes): The axes to draw the signal with, if it is None, cancel the drawing.
        cmap (mpl.colors.LinearSegmentedColormap): The color map.

        dst_cov (np.Array, optional): The covariance matrix being transformed to, if it is None it refers no transformation is produced. Default to None.
        zero_mu_flag (bool, optional): Whether to use 0.0 value as the mu.
        legend_flag (bool, optional): Whether to draw the legend. Defaults to True.

    Returns:
        noise: The generated noise.
    """

    # Make the mu,
    # if not use_mu_flag, zero the mu
    mu = np.repeat(signal[np.newaxis, :], RANDOM_POPULATION, axis=0)

    if zero_mu_flag:
        mu *= 0

    # Generate the noise with random noise with I covariance matrix
    noise = mu + np.random.randn(RANDOM_POPULATION, POINTS) * sigma

    if dst_cov is not None:
        noise = transform_to_cov(noise, dst_cov)

    # Compute the distance of the data between the signal
    # and sort the data by the distance,
    # the order is farthest first
    distance = [np.linalg.norm(d - signal) for d in noise]

    # Sort the data by distance
    noise, distance = sort_by(noise, distance, reverse=True)

    # If ax is not provided, cancel the display
    if ax is None:
        return noise

    # Automatic determines vmin and vmax values
    norm = mpl.colors.Normalize(vmin=np.min(distance), vmax=np.max(distance))
    for y, d in zip(noise, distance):
        ax.plot(y, color=opacity(cmap(norm(d)), 0.5))

    # The latest drawn line is on the most top
    ax.plot(signal, color='red', label='Signal')

    # Place the colorbar
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    # Place the legend
    if legend_flag:
        ax.legend(loc='lower right')

    # Set facecolor and title
    ax.set_facecolor('gray')

    if dst_cov is not None:
        ax.set_title('Signal and its noise: (specific cov)')
    else:
        ax.set_title('Signal and its noise: (sigma={})'.format(sigma))

    return noise


def matprod(*mats: iter):
    """
    Prod the mats using np.matmul for every matrix in mats.

    Args:
        mats (iterable): The list of matrixes to be produced one-by-one.

    Returns:
        output: The production result.
    """
    output = mats[0]
    for m in mats[1:]:
        output = np.matmul(output, m)
    return output


def plot_cov(cov: np.array, suptitle='No title specified'):
    """
    Plot the covariance matrix,
    and compute the svd decomposition of the covariance matrix.

    The decomposition satisfies:

    1. u * s * v^T = cov
    2. u^T * u = I
    3. v^T * v = I

    The eigenvalues are stored in s as (n,) shaped array.

    Args:
        cov (np.array): The covariance matrix in (n x n) shape.
        suptitle (str, optional): The suptitle of the figure. Defaults by 'No title specified'.

    Returns:
        u, s, v: The output of the svd decomposition.
    """

    u, s, v = np.linalg.svd(cov)
    v = v.transpose()

    usvt = matprod(u, np.diag(s), v.transpose())
    utu = np.matmul(u.transpose(), u)
    vtv = np.matmul(v.transpose(), v)

    print('The usvt = cov check (=0?):', np.max(np.abs(usvt - cov)))
    print('The utu  = I   check (=0?):', np.max(np.abs(utu - np.eye(len(cov)))))
    print('The vtv  = I   check (=0?):', np.max(np.abs(vtv - np.eye(len(cov)))))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    ax = axs[0]

    # Ax of the covariance matrix
    im = ax.imshow(cov)

    # Flip the y-axis
    ax.invert_yaxis()

    # Convert the bottom label to the top
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.colorbar(im, shrink=0.5)
    ax.set_title('Covariance matrix')

    # Ax of the eigenvalues
    ax = axs[1]
    ax.plot(s)
    ax.grid(True)
    ax.set_title('Singular values')

    fig.suptitle(suptitle)

    fig.tight_layout()
    # plt.show()

    return u, s, v


def transform_to_cov(x: np.array, cov: np.array, normalize_flag=True):
    """
    Transform x into new_x, the new_x satisfies its covariance matrix is cov.

    The shape of x is (n x m), where n is the number of repeat and m is the dimension.
    The shape of cov is (m x m), the target covariance matrix of the x.

    Args:
        x (np.array): The input signal matrix.
        cov (np.array): The target covariance matrix.
        normalize_flag (bool, optional): Whether normalize the signals. Defaults to True.

    Returns:
        right: The transformed x matrix.
    """

    cov0 = np.matmul(x.transpose(), x)
    u0, s0, v0 = plot_cov(cov0, suptitle='Src cov')
    u, s, v = plot_cov(cov, suptitle='Dst cov')

    eps = 1e-5
    psi = np.sqrt(s / s0)
    psi[s0 < eps] = 0

    def plot_eigenvalues_comparison(s0, s, psi):
        """
        Plot the comparison the eigenvalues of s0 (src) and s (dst).
        The psi is the ratio of s / s0

        Args:
            s0 (1D array): The src eigenvalue vector.
            s (1D array): The dst eigenvalue vector.
            psi (1D array): The ratio of s / s0.
        """

        fig, ax = plt.subplots(figsize=(6, 3))
        ax1 = ax.twinx()

        art_src, = ax.plot(s0 / np.max(s0), color='darkred',
                           label='Src eigenvalue, scaled by: {:.2f}'.format(np.max(s0)))
        art_dst, = ax.plot(s / np.max(s), color='darkblue',
                           label='Dst eigenvalue, scaled by: {:.2f}'.format(np.max(s)))

        art_psi, = ax1.plot(psi, color='black', label='Ratio of Dst. / Src.')

        ax.legend(handles=[art_dst, art_src, art_psi])
        ax.set_ylabel('Scaled eigenvalue')
        ax.set_xlabel('Dimension')
        ax1.set_ylabel('Ratio')
        ax.grid(True)
        ax.set_title('Eigenvalue comparison')

        fig.tight_layout()

        psi = np.diag(psi)

        # plt.show()

    plot_eigenvalues_comparison(s0, s, psi)

    psi = np.diag(psi)

    left = matprod(u, psi, u0.transpose(), x.transpose())
    right = matprod(x, v0, psi, v.transpose())

    # Normalize each noise
    if normalize_flag:
        for s in right:
            s -= np.mean(s)
            s /= np.std(s)

    plot_cov(np.matmul(left, right), suptitle='Outcome cov')

    return right


def signal2cov(signal: np.array):
    """
    Compute the covariance matrix for the signal.

    Args:
        signal (np.array): The signal, it is a 1-D array.

    Returns:
        cov: The covariance matrix.
    """
    cov = np.zeros((POINTS, POINTS))
    for i in range(POINTS):
        for j in range(i, POINTS):
            cov[i, j] = signal[i] - signal[j]
            cov[j, i] = cov[i, j]
    cov *= cov
    cov = np.exp(-cov)
    return cov


# %% ---- 2023-05-24 ------------------------
# Play ground
signal = generate_signal(display_flag=True)
cov = signal2cov(signal)
plot_cov(cov)

# %%

# Draw the signal with its mu
fig, axs = plt.subplots(3, 1, figsize=(6, 6))
generate_noise_for_signal(signal, 1.0, axs[0], CMAP, legend_flag=False)
generate_noise_for_signal(signal, 0.5, axs[1], CMAP, legend_flag=False)
generate_noise_for_signal(signal, 0.1, axs[2], CMAP, legend_flag=True)
fig.tight_layout()
plt.show()

# Draw the signal without its mu
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
noise = generate_noise_for_signal(
    signal, 1.0, ax, CMAP, zero_mu_flag=True)
fig.tight_layout()
plt.show()

# The shape is 50 x 100
noise.shape

# %%
# The output shape is 50 x 100
transform_to_cov(noise, cov)

# %%
# The transformed noise
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
transformed_noise = generate_noise_for_signal(
    signal, 1.0, ax, CMAP, zero_mu_flag=True, dst_cov=cov)
fig.tight_layout()
plt.show()

# %% ---- 2023-05-24 ------------------------
# Pending
