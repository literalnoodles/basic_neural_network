from typing import Union
from typing import Tuple
import numpy as np
import pickle
import gzip

def one_hot_encode(y):
    y_1hot = np.zeros((y.size, 10))
    y_1hot[np.arange(y.size), y] = 1
    return y_1hot

def load_data() -> Tuple[np.ndarray]:
    f = gzip.open('./data_set/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    # one hot encode for training_data
    training_data = (training_data[0], one_hot_encode(training_data[1]))
    f.close()
    return training_data, validation_data, test_data

def shuffle(data_set: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
    X, y = data_set
    feature_size = X.shape[1]
    c = np.c_[X, y]
    np.random.shuffle(c)
    return (c[:, :feature_size], c[:, feature_size:])

def len_data(x: Union[np.ndarray, Tuple[np.ndarray]]) -> int:
    """
    It takes a tuple of numpy arrays or a single numpy array as input and returns the length of the
    first dimension of the array
    
    :param x: the input data
    :type x: Union(np.ndarray, Tuple[np.ndarray])
    :return: The length of the data.
    """
    if (type(x) == tuple):
        return x[0].shape[0]
    return x.shape[0]

def printProg(index, total, length=30, update_rate=1, prefix='', postfix='', endWith='\r'):
    index += 1
    if (index % update_rate != 0):
        return
    if (not postfix):
        postfix = f'{(index / total * 100):.2f}%'
    unit = total / length
    complete = int(index / unit)
    print(f"\r|{complete * '>'}{(length - complete) * '-'}| {postfix}", end=endWith)
    if (index == total):
        print('\r' +' ' * (4 + length + len(postfix) + len(prefix)), end='\r')