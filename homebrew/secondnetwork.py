import network as net
import numpy as np
import os, sys
import mnist_io
# import mnist_view

def to_categorical(ndarray, classes: int) -> np.ndarray:
    return np.array([[0 if val!=i else 1 for i in range(classes)] for val in ndarray])

def from_categorical(ndarray: np.ndarray) -> np.ndarray:
    outputs = np.empty(0)
    for arr in ndarray:
        outputs = np.append(outputs, (arr == max(arr)).nonzero())
    return outputs

if __name__ == '__main__':
    network = net.JIT_Network(input_shape=784, output_shape=10, node_count=784+512+10, learning_rate=0.00001)

    dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "datasets")

    for i in range(0, 784):
        for j in range(784, 784+512):
            network.add_connection(j, i)
    for i in range(784, 784+512):
        for j in range(784+512, 784+512+10):
            network.add_connection(j, i)
    print("Connecting completed")

    set_count = 500

    train_images = mnist_io.images_from_file(os.path.join(dataset, "train-images-idx3-ubyte/train-images.idx3-ubyte"), set_count)
    train_images = train_images.reshape(set_count, 784).astype('float32')
    train_images /= 255

    train_labels = mnist_io.labels_from_file(os.path.join(dataset, "train-labels-idx1-ubyte/train-labels.idx1-ubyte"), set_count)

    test_images = mnist_io.images_from_file(os.path.join(dataset, "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"), set_count)
    test_images = test_images.reshape(set_count, 784).astype('float32')
    test_images /= 255

    test_labels = mnist_io.labels_from_file(os.path.join(dataset, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), set_count)

    # view = mnist_view.ViewData(train_images.reshape(set_count, 28, 28), train_labels)

    network.train(train_images, to_categorical(train_labels, 10), 2)

    print(from_categorical(np.array(network.predict(test_images[:5]))), "===", test_labels[:5])

   # 2:34 Stack()
   # 2:18 []
   # microoptimization