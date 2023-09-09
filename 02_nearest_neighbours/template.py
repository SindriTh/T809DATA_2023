# Author: Sindri Þór Harðarson
# Date: 26.08.2023
# Project: 02 Nearest Neighbor
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points
from help import remove_one

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.sqrt(np.sum((x - y)**2))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances    


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    return np.argsort(distances)[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    votes = np.zeros(len(classes))
    for target in targets:
        votes[target] += 1
    return np.argmax(votes)


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest = k_nearest(x, points, k)
    return vote(point_targets[nearest], classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Apply knn to all points to predict a class label for each point
    '''
    predictions = np.zeros(point_targets.shape[0])
    for i in range(point_targets.shape[0]):
        # Remove the i-th point from points and point_targets
        updated_points = remove_one(points, i)
        updated_targets = remove_one(point_targets, i)

        # Make a prediction for the i-th point
        predictions[i] = knn(points[i], updated_points, updated_targets, classes, k)

    return predictions.astype(int)


# Expected [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 1 1 1 1 1 2 0 1 1 1]
# Actual   [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 1 1 1 2 1 2 0 1 1 1]

# Expected [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 2 1 1 2 1 2 0 1 1 2]
# Actual   [2 2 2 2 0 1 0 1 1 0 1 2 1 2 2 0 1 0 2 2 1 1 2 1 2 0 1 1 2]


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    '''
    Calculate the accuracy of knn
    '''
    predictions = knn_predict(points, point_targets, classes, k)
    return np.sum(predictions == point_targets) / point_targets.shape[0]


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Calculate the confusion matrix of knn
    '''
    # Initialize the confusion matrix with zeros
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Get the knn predictions for all points
    predictions = knn_predict(points, point_targets, classes, k)

    # Loop through each (prediction, actual) pair and update the confusion matrix
    for pred, actual in zip(predictions, point_targets):
        confusion_matrix[pred, actual] += 1
    
    return confusion_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    '''
    Find the best k for knn
    '''
    highest_accuracy = 0
    best_k = 0
    for k in range(1, points.shape[0]):
        accuracy = knn_accuracy(points, point_targets, classes, k)
        if(accuracy > highest_accuracy):
            highest_accuracy = accuracy
            best_k = k
    return best_k
    

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    ''' Similar to plot_points but if the point is correct it is green otherwise red'''
    colors = ['yellow', 'purple', 'blue']
    edgecolors = ['red', 'green']
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        edge = edgecolors[int(point_targets[i] == knn(points[i], points, point_targets, classes, k))]
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edge,
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.savefig('plots/2_5_1.png')
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    num_classes = len(classes)
    weighted_votes = np.zeros(num_classes)
    
    for target, distance in zip(targets, distances):
        if distance == 0:
            return target  # If the distance is zero, immediately return this class as it's the exact same point.
        weight = 1.0 / distance
        weighted_votes[target] += weight
    
    return np.argmax(weighted_votes)


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_indices = k_nearest(x, points, k)
    nearest_targets = point_targets[nearest_indices]
    nearest_distances = euclidian_distances(x, points[nearest_indices])
    return weighted_vote(nearest_targets, nearest_distances, classes)


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    predictions = np.zeros(point_targets.shape[0])
    for i in range(point_targets.shape[0]):
        updated_points = remove_one(points, i)
        updated_targets = remove_one(point_targets, i)
        predictions[i] = wknn(points[i], updated_points, updated_targets, classes, k)
    return predictions.astype(int)

def __wknn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    '''
    Calculate the accuracy of weighted kNN (wkNN)
    '''
    predictions = wknn_predict(points, point_targets, classes, k)
    return np.sum(predictions == point_targets) / point_targets.shape[0]


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    n = points.shape[0]
    k_values = range(1, n)
    knn_accuracies = []
    wknn_accuracies = []

    for k in k_values:
        knn_acc = knn_accuracy(points, targets, classes, k)
        wknn_acc = __wknn_accuracy(points, targets, classes, k)  # Replace this with wknn_accuracy when you implement it
        knn_accuracies.append(knn_acc)
        wknn_accuracies.append(wknn_acc)

    plt.plot(k_values, knn_accuracies, label='kNN')
    plt.plot(k_values, wknn_accuracies, label='wkNN')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/b_4_1.png')
    plt.show()


#### THEORETICAL
# The main reason for the difference in performance between kNN and wkNN as 
# k increases is due to how they consider neighborhood points when making decisions. In standard kNN, every neighbor has an equal say in determining the class of a new point, which might not be desirable when the dataset has noise or outliers. As 
# k increases, kNN might include more "distant" neighbors, diluting the influence of the nearest neighbors and potentially reducing performance.
# On the other hand, wkNN assigns weights to neighbors based on their distance to the test point. Closer neighbors have more influence on the classification. This is particularly useful when the value of 
# k is large, as it can effectively "ignore" distant neighbors by assigning them low weights. As a result, wkNN is often more robust to the choice of 
# k, especially in the presence of noise and outliers.


if __name__ == "__main__":
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]
    #plot_points(d, t)
    # print(euclidian_distance(x, points[0]))  # Output should be around 0.5385164807134502
    # print(euclidian_distance(x, points[50]))
    # print(euclidian_distances(x, points))
    # print(k_nearest(x, points, 1))
    # print(k_nearest(x, points, 3))
    # print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    # print(vote(np.array([1,1,1,1]), np.array([0,1])))
    # print(knn(x, points, point_targets, classes, 1))
    # print(knn(x, points, point_targets, classes, 5))
    # print(knn(x, points, point_targets, classes, 150))
    # (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    # print(knn_predict(d_test, t_test, classes, 10))
    # print(knn_predict(d_test, t_test, classes, 5))
    # print(knn_accuracy(d_test, t_test, classes, 10))
    # print(knn_accuracy(d_test, t_test, classes, 5))
    # print(knn_confusion_matrix(d_test, t_test, classes, 10))
    # print(knn_confusion_matrix(d_test, t_test, classes, 20))
    # # print(best_k(d_train, t_train, classes))
    # knn_plot_points(d, t, classes, 3)
    #######
    compare_knns(points, point_targets, classes)
    print()