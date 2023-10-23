# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''

    return np.sqrt(np.sum(np.square(X[:, np.newaxis] - Mu), axis=2))
    

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    r = np.zeros(dist.shape, dtype=int)
    
    # Find the index of the minimum distance
    min_dist_indices = np.argmin(dist, axis=1)
    
    # Set the appropriate positions to 1 based on the min_dist_indices
    r[np.arange(r.shape[0]), min_dist_indices] = 1
    
    return r


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    total_distance = np.sum(R * dist)
    return total_distance / R.shape[0]


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    return np.dot(R.T, X) / np.sum(R, axis=0)[:, np.newaxis]


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []

    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu) # Distance matrix
        R = determine_r(dist) # Indicator matrix
        j = determine_j(R, dist) # Objective function value
        Mu = update_Mu(Mu, X_standard, R) # Update prototypes

        Js.append(j) # Save objective function value


    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


def _plot_j():
    X, y, c = load_iris()
    Mu, R, j = k_means(X, 4, 10)
    plt.figure(figsize=(8, 5))
    plt.plot(j, '-o', color='black')
    plt.title("Objective Function Progression")
    plt.xlabel("Iteration")
    plt.ylabel("$\hat{J}$ Value")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("figs/1_6_1.png", dpi=300)
    plt.show()


def _plot_multi_j():
    k_values = [2, 3, 5, 10]
    plt.figure(figsize=(10, 6))
    
    for k in k_values:
        _, _, Js = k_means(X, k, 10)
        plt.plot(Js, '-o', label=f'k={k}')
    
    plt.title("Objective Function Progression for Different k Values")
    plt.xlabel("Iteration")
    plt.ylabel("$\hat{J}$ Value")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("figs/1_7_1.png", dpi=300)
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    k = len(classes)
    Mu, R, _ = k_means(X, k, num_its)
    
    # Assign each cluster a label based on majority class in the cluster
    cluster_labels = {}
    for cluster in range(k):
        indices = np.where(R[:, cluster] == 1)[0]
        labels = t[indices]
        cluster_label = np.bincount(labels).argmax()
        cluster_labels[cluster] = cluster_label
    
    # Make predictions based on the cluster labels
    predictions = np.array([cluster_labels[np.argmax(r)] for r in R])
    
    return predictions


def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    predictions = k_means_predict(X, y, c, 10)
    print(accuracy_score(y, predictions))
    print(confusion_matrix(y, predictions))
    


def _my_kmeans_on_image():
# Try first running your own k_means function on the image data with 7 clusters for 5 iterations. You should notice how incredibly slow it is.
# Since our implementation is so slow, maybe we should try using an sklearn implementation, namely sklearn.KMeans.
# Finish implementing the function plot_image_clusters. In your PDF, show plots for num_clusters=2, 5, 10, 20. Name these plots as 2_1_1.png-2_1_4.png.
    image, (w, h) = image_to_numpy('images/clown.png')
    X = image.reshape((-1, 3))
    k = 7
    Mu, R, _ = k_means(X, k, 5)
    segmented_image = np.array([Mu[np.argmax(r)] for r in R]).reshape((w, h, 3))
    plt.imshow(segmented_image.astype(np.uint8))
    plt.show()

    Mu, R, _ = KMeans(n_clusters=k, max_iter=5)
    segmented_image = np.array([Mu.cluster_centers_[np.argmax(r)] for r in R]).reshape((w, h, 3))
    plt.imshow(segmented_image.astype(np.uint8))
    plt.show()
    


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image)

    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


if __name__ == '__main__':

    # Section 1.1
    a = np.array([
        [1, 0, 0],
        [4, 4, 4],
        [2, 2, 2]])
    b = np.array([
        [0, 0, 0],
        [4, 4, 4]])
    # print(distance_matrix(a, b))
    # [[1, 6.40312424], [6.92820323, 0], [3.46410162, 3.46410162]]
    assert(np.allclose(distance_matrix(a, b), [[1, 6.40312424], [6.92820323, 0], [3.46410162, 3.46410162]]))

    # Section 1.2
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    # print(determine_r(dist))
    assert (determine_r(dist) == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]).all()

    # Section 1.3
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    R = determine_r(dist)
    # print(determine_j(R, dist))
    assert(determine_j(R,dist) == 0.9)

    # Section 1.4
    X = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]])
    Mu = np.array([
        [0.0, 0.5, 0.1],
        [0.8, 0.2, 0.3]])
    R = np.array([
        [1, 0],
        [0, 1],
        [1, 0]])
    # print(update_Mu(Mu, X, R))
    assert(update_Mu(Mu, X, R) == [[0,0.5,0],[1,0,0]]).all()

    # Section 1.5
    X, y, c = load_iris()
    k_means(X, 4, 10)
    assert k_means(X, 4, 10)[0].shape == (4, 4)
    assert k_means(X, 4, 10)[1].shape == (150, 4)
    assert len(k_means(X, 4, 10)[2]) == 10
    # print(k_means(X, 4, 10)[0][0:2])
    # print(k_means(X, 4, 10)[1][0:2])
    # print(k_means(X, 4, 10)[2][0:2])

    # Section 1.6
    # _plot_j()

    # Section 1.7
    # _plot_multi_j()

    # Section 1.8
    # _iris_kmeans_accuracy()

    # def my_multi_j():
    #     k_values = [2, 3, 5, 10, 15, 20, 25, 50, 75, X.shape[0]]
    #     plt.figure(figsize=(10, 6))
        
    #     for k in k_values:
    #         _, _, Js = k_means(X, k, 10)
    #         print(f"k={k}: {Js[-1]}")
    #         plt.plot(k,Js[-1], '-o', label=f'k={k}')
        
    #     plt.title("Objective Function Progression for Different k Values")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("$\hat{J}$ Value")
    #     plt.legend()
    #     plt.grid(True, which="both", linestyle="--")
    #     plt.tight_layout()
    #     plt.show()
    #     for k in k_values:
    #         _, _, Js = k_means(X, k, 10)
    #         print(f"k={k}: Js={Js}")
    #         plt.plot(Js, '-o', label=f'k={k}')
        
    #     plt.title("Objective Function Progression for Different k Values")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("$\hat{J}$ Value")
    #     plt.legend()
    #     plt.grid(True, which="both", linestyle="--")
    #     plt.tight_layout()
    #     plt.show()
        
    # Print to make sure everything works
    # my_multi_j()
    # _my_kmeans_on_image()
    plot_image_clusters(2)
    plot_image_clusters(5)
    plot_image_clusters(10)
    plot_image_clusters(20)
    print("-----------------")
    print("Everything Works!")