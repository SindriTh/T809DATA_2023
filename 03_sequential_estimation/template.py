# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    cov_matrix = np.eye(k) * var ** 2 # Identity matrix * var
    return np.random.multivariate_normal(mean, cov_matrix, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + ((1.0 / n) * (x - mu))


def _plot_sequence_estimate():
    np.random.seed(1234)
    data = gen_data(100, 3, np.array([0, 0, 0]), 4)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[-1], data[i], i + 1))

    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    
    plt.title('Rolling Estimate of Mean')
    plt.xlabel('Number of Data Points')
    plt.legend(loc='upper center')
    plt.savefig('figures/1_5_1.png', dpi=300)
    plt.show()


def _square_error(y, y_hat):
    return (y - y_hat) ** 2



def _plot_mean_square_error():
  np.random.seed(1234)
  data = gen_data(100, 3, np.array([0, 0, 0]), 4)
  estimates = [np.array([0, 0, 0])]
  errors = []

  for i, datapoint in enumerate(data):
      estimates.append(update_sequence_mean(estimates[-1], datapoint, i + 1))
      errors.append(np.mean(_square_error(np.mean([data]), estimates[-1])))


  plt.plot(errors)
  plt.title('Mean Square Error')
  plt.xlabel('Number of Data Points')
  plt.ylabel('Mean Square Error')
  plt.savefig('figures/1_6_1.png', dpi=300)
  plt.show()



# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass



if __name__ == "__main__":

    # 1.1 Generate data
    np.random.seed(1234)
    print(gen_data(2, 3, np.array([0,1,-1]), 1.3))
    np.random.seed(1234)
    print(gen_data(5, 1, np.array([0.5]), 0.5))

    # 1.2 Visualization
    np.random.seed(1234)
    X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    # Visualization for 1.2
    #scatter_3d_data(X)
    #bar_per_axis(X)
    
    #1.4
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    print(update_sequence_mean(mean, new_x, X.shape[0]))

    #1.5
    _plot_sequence_estimate()
    _plot_mean_square_error()    
