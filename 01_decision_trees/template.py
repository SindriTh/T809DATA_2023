# Author: Sindri Þór Harðarson
# Date: 26.08.2023
# Project: 01 Decision Trees
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    N = len(targets)  # Total number of data points
    priors = []  # List to store the prior probabilities
    targets = list(targets)  # Convert targets to a list
    # Loop over each class to calculate its prior probability
    for c in classes:
        count = targets.count(c)  # Count the occurrence of class c in targets
        priors.append(count / N)  # Calculate the prior probability and append it to the list
    
    return np.array(priors)



def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1, targets_1, features_2, targets_2 = [], [], [], []

    # Loop over each sample in the data set
    for i in range(len(features)):
        # Check if the feature value is less than theta
        if features[i][split_feature_index] < theta:
            # Append the feature and target to the first data set
            features_1.append(features[i])
            targets_1.append(targets[i])
        else:
            # Append the feature and target to the second data set
            features_2.append(features[i])
            targets_2.append(targets[i])
        
    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    # Calculate the prior probabilities for each class
    priors = prior(targets, classes)
    
    # Calculate the sum of squared probabilities
    sum_squared_priors = sum([p ** 2 for p in priors])
    
    # Calculate the Gini impurity using the formula
    impurity = 0.5 * (1 - sum_squared_priors)
    
    return impurity

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    t1 = np.array(t1) if isinstance(t1, list) else t1
    t2 = np.array(t2) if isinstance(t2, list) else t2
    n = t1.shape[0] + t2.shape[0]
    
    weighted_gini = (t1.shape[0] * g1 / n) + (t2.shape[0] * g2 / n)

    return weighted_gini

def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    
    # Split the data using the given feature index and threshold
    (_, targets_1), (_, targets_2) = split_data(features, targets, split_feature_index, theta)
    
    # Convert lists to numpy arrays
    targets_1 = np.array(targets_1)
    targets_2 = np.array(targets_2)

    # Calculate the weighted Gini impurity for the split
    weighted_gini = weighted_impurity(targets_1, targets_2, classes)
    
    return weighted_gini

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min_val = np.min(features[:, i])
        max_val = np.max(features[:, i])
        thetas = np.linspace(min_val, max_val, num_tries + 2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)
        

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree, filled=True)
        plt.savefig('plots/2_3_1.png', dpi=300)
        plt.show()

    def plot_progress(self):
        # Independent section
        accuracies = []  # To store accuracies at each step
        sample_sizes = list(range(1, len(self.train_features) + 1))  # From one sample to all samples
        
        for i in sample_sizes:
            # Use only the first i samples from the training set
            train_subset_features = self.train_features[:i]
            train_subset_targets = self.train_targets[:i]
            
            # Train the tree
            self.tree.fit(train_subset_features, train_subset_targets)
            
            # Evaluate on the test set and store the accuracy
            acc = self.tree.score(self.test_features, self.test_targets)
            accuracies.append(acc)
        
        # Create the plot
        plt.figure()
        plt.plot(sample_sizes, accuracies)
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy on test set')
        plt.title('Accuracy vs Number of training samples')
        plt.savefig('plots/indep_1.png')  # Save the plot
        plt.show()

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        # Initialize the confusion matrix
        confusion_matrix = np.zeros((len(self.classes), len(self.classes)))

        for i in range(len(self.test_features)):
            # Get the predicted class and the true class
            predicted_class = self.guess()[i]
            true_class = self.test_targets[i]

            # Increment the confusion matrix
            confusion_matrix[predicted_class][true_class] += 1

        return confusion_matrix

if __name__ == "__main__":
    print(prior([0,0,1],[0,1]))
    print(prior([0,2,3,3],[0,1,2,3]))

    features, targets, classes = load_iris()
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    print(gini_impurity(t_1, classes))
    print(gini_impurity(t_2, classes))
    print(weighted_impurity(t_1, t_2, classes))
    print(total_gini_impurity(features, targets, classes, 2, 4.65))
    print(brute_best_split(features, targets, classes, 30))

    dt = IrisTreeTrainer(features, targets, classes=classes)
    dt.train()
    print(f'The accuracy is: {dt.accuracy()}')
    dt.plot()
    print(f'I guessed: {dt.guess()}')
    print(f'The true targets are: {dt.test_targets}')
    print(dt.confusion_matrix())

    dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    dt.plot_progress()
