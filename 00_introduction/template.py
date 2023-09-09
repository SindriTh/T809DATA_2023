import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    const = 1 / np.sqrt(2 * np.pi * np.power(sigma, 2))
    exponent = np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
    return const * exponent

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    x = np.linspace(x_start, x_end, 500)
    y = normal(x, sigma, mu)
    plt.plot(x, y)

def _plot_three_normals():
    plt.figure()
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.savefig("./plots/1_2_1.png")
    #plt.show()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    p = 0
    for sigma, mu, weight in zip(sigmas, mus, weights):
        p += weight * normal(x, sigma, mu)
    return p

def _compare_components_and_mixture():
    plt.figure()
    x_vals = np.linspace(-5, 5, 500)

    # Plot individual normals
    plot_normal(0.5, 0, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_normal(0.25, 1.5, -5, 5)

    # Plot mixture
    y_vals = normal_mixture(x_vals, [0.5, 1.5, 0.25], [0, -0.5, 1.5], [1/3, 1/3, 1/3])
    plt.plot(x_vals, y_vals, label='Mixture')

    plt.legend()
    plt.savefig("./plots/2_2_1.png")


def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    counts = np.random.multinomial(n_samples, weights)
    samples = []

    for count, sigma, mu in zip(counts, sigmas, mus):
        samples.extend(np.random.normal(mu, sigma, count))

    return np.array(samples)


def plot_mixture_and_samples():
    plt.figure(figsize=(16, 4))

    mus = [0, -1, 1.5]
    sigmas = [0.3, 0.5, 1]
    weights = [0.2, 0.3, 0.5]

    for idx, n_samples in enumerate([10, 100, 500, 1000]):
        plt.subplot(1, 4, idx + 1)
        samples = sample_gaussian_mixture(sigmas, mus, weights, n_samples)

        plt.hist(samples, 100, density=True)

        x_vals = np.linspace(-5, 5, 500)
        y_vals = normal_mixture(x_vals, sigmas, mus, weights)
        plt.plot(x_vals, y_vals, label='Mixture')

    plt.savefig("./plots/3_2_1.png")


if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    print(normal(0, 1, 0))
    print(normal(3, 1, 5))
    print(normal(np.array([-1,0,1]), 1, 0))
    #plt.figure()
    #plot_normal(0.5, 0, -2, 2)
    _plot_three_normals()
    print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))
    print(normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1]))
    print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3))
    print(sample_gaussian_mixture([0.1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10))
    plot_mixture_and_samples()
    