from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    return 1/(1+np.exp(-x))
    


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x)*(1-sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    
    return np.dot(x,w), sigmoid(np.dot(x,w))


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x,0,1)
    a1 = np.dot(z0,W1)
    z1 = np.insert(sigmoid(a1),0,1)
    a2 = np.dot(z1,W2)
    y = sigmoid(a2)
    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    deltak = y - target_y
    deltaj = np.dot(W2[1:,:],deltak)*d_sigmoid(a1)
    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)
    dE1 += np.outer(z0,deltaj)
    dE2 += np.outer(z1,deltak)
    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''

    N = X_train.shape[0]
    
    Etotal = np.zeros(iterations)
    misclassification_rate = np.zeros(iterations)

    for i in range(iterations):
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)
        last_guesses = []
        misclassification_count = 0

        for j in range(N):
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0
            y, dE1, dE2 = backprop(X_train[j], target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2
            # Cross entropy error
            Etotal[i] += np.sum(-target_y*np.log(y) - (1-target_y)*np.log(1-y))
            last_guesses.append(np.argmax(y))
            if np.argmax(y) != t_train[j]:
                misclassification_count += 1
            # misclassification_rate[i] += np.sum(np.abs(np.round(y)-target_y))

        W1 = W1 - eta*dE1_total / N
        W2 = W2 - eta*dE2_total / N
        Etotal[i] = Etotal[i] / N
        misclassification_rate[i] = misclassification_count / N

    return W1, W2, Etotal, misclassification_rate, last_guesses


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    guesses = []
    for i in range(X.shape[0]):
        y, _, _, _, _ = ffnn(X[i], M, K, W1, W2)
        guesses.append(np.argmax(y))
    return np.array(guesses)


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
    # Section 1.1
    print(sigmoid(0.5))
    print(d_sigmoid(0.2))

    # Section 1.2
    print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    print(perceptron(np.array([0.2,0.4]), np.array([0.1,0.4])))

    # Section 1.3
    np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
        split_train_test(features, targets)
    np.random.seed(1234)
    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    print(y)
    print(z0)
    print(z1)
    print(a1)
    print(a2)

    # Section 1.4
    np.random.seed(42)
    M = 6
    D = train_features.shape[1]
    x = features[0, :]

    target_y = np.zeros(K)
    target_y[targets[0]] = 1.0

    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print(y)
    print(dE1)
    print(dE2)

    # Section 2.1
    print( "Section 2.1" )
    np.random.seed(1231)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    print('W1tr')
    print(W1tr)
    print('W2tr')
    print(W2tr)
    print('ETOT')
    print(Etotal)
    print('MISC')
    print(misclassification_rate)
    print('LAST')
    print(last_guesses)

    def test_accuracy(guesses, train_targets):
        confusion_matrix = np.zeros((K,K))
        for i in range(len(guesses)):
            confusion_matrix[train_targets[i],guesses[i]] += 1
        print(confusion_matrix)
        # test accuracy
        correct = 0
        for i in range(len(guesses)):
            if guesses[i] == train_targets[i]:
                correct += 1
        print(correct/len(guesses))
        # import matplotlib.pyplot as plt
        # plt.plot(Etotal)
        # plt.ylabel('E_total')
        # plt.xlabel('iterations')
        # plt.savefig('figs/Etotal.png', dpi=300)
        # plt.show()
        # plt.plot(misclassification_rate)
        # plt.ylabel('misclassification_rate')
        # plt.xlabel('iterations')
        # plt.savefig('figs/misclassification_rate.png', dpi=300)
        # plt.show()

        
    # Section 2.2
    
    # Section 2.3
    print( "Section 2.2" )
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features, train_targets, M, K, W1, W2, 500, 0.1)
    X_test = test_features
    guesses = test_nn(X_test, M, K, W1tr, W2tr)
    print(guesses)
    test_accuracy(guesses, test_targets)