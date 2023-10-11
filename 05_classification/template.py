# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    classes = features[targets == selected_class]
    return np.mean(classes, axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    classes = features[targets == selected_class]
    return np.cov(classes, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal.pdf(feature, mean=class_mean, cov=class_covar)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([likelihood_of_class(test_features[i], means[j], covs[j]) for j in classes])
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs, priors = [], [], []
    total_samples = len(train_targets)
    
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
        priors.append(np.sum(train_targets == class_label) / total_samples)
    
    aposterioris = []
    for i in range(test_features.shape[0]):
        sample_aposterioris = [likelihood_of_class(test_features[i], means[j], covs[j]) * priors[j] for j in classes]
        aposterioris.append(sample_aposterioris)
    
    return np.array(aposterioris)



if __name__ == "__main__":
    def confusion_matrix(classes, test_features, test_targets, guess):
        # Initialize the confusion matrix
        confusion_matrix = np.zeros((len(classes), len(classes)))

        for i in range(len(test_features)):
            # Get the predicted class and the true class
            predicted_class = guess()[i]
            true_class = test_targets[i]

            # Increment the confusion matrix
            confusion_matrix[predicted_class][true_class] += 1

        return confusion_matrix
    
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)
    print("Section 1.1: ")
    print(mean_of_class(train_features, train_targets, 0))
    print("Section 1.2: ")
    print(covar_of_class(train_features, train_targets, 0))
    print("Section 1.3: ")
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    print(likelihood_of_class(train_features[0, :], class_mean, class_cov))
    print("Section 1.4: ")
    print(maximum_likelihood(train_features, train_targets, test_features, classes))
    print("Section 1.5: ")
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(predict(likelihoods))
    print("Section 2.1: ")
    print(maximum_aposteriori(train_features, train_targets, test_features, classes))
    print("Section 2.2: ")
    # Confusion matrices for maximum likelihood and maximum a posteriori
    likelihoods_ml = maximum_likelihood(train_features, train_targets, test_features, classes)
    predictions_ml = predict(likelihoods_ml)

    aposterioris_map = maximum_aposteriori(train_features, train_targets, test_features, classes)
    predictions_map = predict(aposterioris_map)

    # 2. Compute the Accuracy
    accuracy_ml = np.mean(predictions_ml == test_targets)
    accuracy_map = np.mean(predictions_map == test_targets)

    print(f"Accuracy of Maximum Likelihood: {accuracy_ml}")
    print(f"Accuracy of Maximum A Posteriori: {accuracy_map}")

    # 3. Generate Confusion Matrices
    def generate_confusion_matrix(predictions, test_targets, classes):
        matrix = np.zeros((len(classes), len(classes)))
        for i in range(len(predictions)):
            matrix[predictions[i]][test_targets[i]] += 1
        return matrix

    confusion_ml = generate_confusion_matrix(predictions_ml, test_targets, classes)
    confusion_map = generate_confusion_matrix(predictions_map, test_targets, classes)

    print("Confusion Matrix for Maximum Likelihood:")
    print(confusion_ml)
    print("Confusion Matrix for Maximum A Posteriori:")
    print(confusion_map)