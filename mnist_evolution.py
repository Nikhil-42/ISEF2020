from evolutionary_module import evolution
import numpy as np
import numba
import mnist_io
import os

@numba.njit
def to_categorical(ndarray, classes: int) -> np.ndarray:
    return np.array([[0 if val!=i else 1 for i in range(classes)] for val in ndarray])

@numba.njit
def from_categorical(ndarray: np.ndarray) -> np.ndarray:
    outputs = np.empty(0)
    for arr in ndarray:
        outputs = np.append(outputs, (arr == max(arr)).nonzero())
    return outputs.astype(int)

dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "Python\\datasets")

train_images = mnist_io.images_from_file(os.path.join(dataset, "train-images-idx3-ubyte\\train-images.idx3-ubyte"), 60000)
train_images = train_images.reshape(60000, 784).astype('float32')
train_images /= 255

train_labels = to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "train-labels-idx1-ubyte\\train-labels.idx1-ubyte"), 60000), 10)

test_images = mnist_io.images_from_file(os.path.join(dataset, "t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"), 10000)
test_images = test_images.reshape(10000, 784).astype('float32')
test_images /= 255

test_labels = to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), 10000), 10)

@numba.njit
def categorical_compare(output_layer, expected):
    return output_layer.argsort()[-1]==expected.argsort()[-1]

evolution(train_images, train_labels, test_images, test_labels, categorical_compare, population_size=30, population_count=1, node_cap=2000, r_seed=12)