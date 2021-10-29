import time
import numpy as np


def sigmod(y_pred):
    return np.array([_sigmoid(x) for x in y_pred])


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x)) # avoid overflow runtime error


def preSort(X):
    n_samples, n_features = np.shape(X)
    block = None

    for i in range(n_features):

        index = np.array(range(0, n_samples, 1))
        index = index.reshape(len(index), 1)

        feature = X[:, i:i + 1]
        feature_index = np.concatenate((feature, index), axis=1)
        feature_index_sort = feature_index[np.argsort(feature_index[:, 0])]

        index = feature_index_sort[:, 1:2]

        if block is None:
            block = index
        else:
            block = np.concatenate((block, index), axis=1)

    return block


def timecost(text=""):
    """用于测量函数用时的装饰器工具"""

    def decorator(func):
        def wrapper(*args, **kw):
            begin = time.time()
            result = func(*args, **kw)
            end = time.time()
            print(text, end - begin)
            return result

        return wrapper

    return decorator