# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_standardized = standardize(X)
    plt.scatter(X_standardized[:, i], X_standardized[:, j])

def _scatter_cancer():
    X, y = load_cancer()
    plt.figure(figsize=(15, 10))
    for idx in range(30):
        plt.subplot(5, 6, idx+1)
        scatter_standardized_dims(X, 0, idx)
    plt.savefig('figures/1_3_1.png', dpi=300)


def _plot_pca_components():
    X, y = load_cancer()
    X_standardized = standardize(X)
    pca = PCA()
    pca.fit_transform(X_standardized)
    components = pca.components_
    
    plt.figure(figsize=(15, 10))
    for idx in range(components.shape[0]):
        plt.subplot(5, 6, idx+1)
        plt.title(f'PCA {idx+1}')
        plt.plot(components[idx], label=f'PCA {idx+1}')
    plt.savefig('figures/2_1_1.png', dpi=300)
    plt.show()


def _plot_eigen_values():
    X, y = load_cancer()
    X_standardized = standardize(X)
    pca = PCA()
    pca.fit_transform(X_standardized)
    eigenvalues = pca.explained_variance_
    plt.plot(eigenvalues)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig('figures/3_1_1.png', dpi=300)
    plt.show()


def _plot_log_eigen_values():
    X, y = load_cancer()
    X_standardized = standardize(X)
    pca = PCA()
    pca.fit_transform(X_standardized)
    eigenvalues = pca.explained_variance_
    plt.plot(np.log10(eigenvalues))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.savefig('figures/3_2_1.png')
    plt.show()


def _plot_cum_variance():
    X, y = load_cancer()
    X_standardized = standardize(X)
    pca = PCA()
    pca.fit_transform(X_standardized)
    eigenvalues = pca.explained_variance_
    cum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    plt.plot(cum_variance)

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.savefig('figures/3_3_1.png')
    plt.show()

if __name__ == "__main__":
    print("Section 1.1: ")
    print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))
    
    print("Section 1.2: ")
    # X = np.array([
    # [1, 2, 3, 4],
    # [0, 0, 0, 0],
    # [4, 5, 5, 4],
    # [2, 2, 2, 2],
    # [8, 6, 4, 2]])
    # scatter_standardized_dims(X, 0, 2)
    # plt.show()
    
    print("Section 1.3: ")
    # _scatter_cancer()
    # plt.show()
    print("Section 2.1: ")
    # _plot_pca_components()
    print("Section 3.1: ")
    _plot_eigen_values()
    print("Section 3.2: ")
    _plot_log_eigen_values()
    print("Section 3.3: ")
    _plot_cum_variance()
