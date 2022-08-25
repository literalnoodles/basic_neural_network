from matplotlib import pyplot
import pickle
import gzip
import numpy as np
def load_data():
    f = gzip.open('./data_set/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    training_x = [x.reshape(-1, 1) for x in training_data[0]]
    training_y = [vectorize(y) for y in training_data[1]]
    validation_x = [x.reshape(-1, 1) for x in validation_data[0]]
    test_data_x = [x.reshape(-1, 1) for x in test_data[0]]
    training_data = list(zip(training_x, training_y))
    validation_data = list(zip(validation_x, validation_data[1]))
    test_data = list(zip(test_data_x, test_data[1]))
    f.close()
    return training_data, validation_data, test_data

def vectorize(x):
    vector_x = np.zeros((10, 1))
    vector_x[x] = 1.0
    return vector_x

def show_result(image_data, result_data):
    for i in range(10):
        pyplot.subplot(4, 4, i + 1)
        pyplot.title(result_data[i])
        pyplot.axis('off')
        pyplot.imshow(image_data[i].reshape(-1, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.tight_layout(pad=1.5)
    pyplot.axis('off')
    pyplot.show()

