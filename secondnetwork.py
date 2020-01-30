import homebrew.network as net
import numpy as np
from numba import njit
import os
import mnist_io
# import mnist_view

@njit
def to_categorical(ndarray, classes: int) -> np.ndarray:
    return np.array([[0 if val!=i else 1 for i in range(classes)] for val in ndarray])

@njit
def from_categorical(ndarray):
    outputs = np.zeros(ndarray.shape[0])
    for i, arr in enumerate(ndarray):
        outputs[i] = arr.argsort()[-1]
    return outputs.astype(int)

if __name__ == '__main__':
    hidden_nodes = 150
    network = net.JIT_Network(input_shape=784, output_shape=10, node_count=784+hidden_nodes+10, learning_rate=0.005, id_num=0)

    dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "Python\\datasets")

    # tos = net.gen_repeated_range(0, 10, 2)
    # print([to for to in tos])
    for i in range(0, 784):
        for j in range(784, 784+hidden_nodes):
            network.add_connection(j, i)

    for i in range(784, 784+hidden_nodes):
        for j in range(784+hidden_nodes, 784+hidden_nodes+10):
            network.add_connection(j, i)
    print("Connecting completed")

    set_count = 60000

    train_images = mnist_io.images_from_file(os.path.join(dataset, "train-images-idx3-ubyte/train-images.idx3-ubyte"), set_count)
    train_images = train_images.reshape(set_count, 784).astype('float32')
    train_images /= 255

    train_labels = mnist_io.labels_from_file(os.path.join(dataset, "train-labels-idx1-ubyte/train-labels.idx1-ubyte"), set_count)

    if set_count > 10000:
        set_count = 10000

    test_images = mnist_io.images_from_file(os.path.join(dataset, "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"), set_count)
    test_images = test_images.reshape(set_count, 784).astype('float32')
    test_images /= 255

    test_labels = mnist_io.labels_from_file(os.path.join(dataset, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), set_count)

    # view = mnist_view.ViewData(train_images.reshape(set_count, 28, 28), train_labels)

    network.train(train_images, to_categorical(train_labels, 10), 1, 2000, 0.01)


    @njit
    def categorical_compare(output_layer, expected):
        return output_layer.argsort()[-1] == expected.argsort()[-1]

    print(network.validate(test_images, to_categorical(test_labels, 10), categorical_compare))

   # 2:34 Stack()
   # 2:18 []
   # microoptimization