from homebrew.network import JIT_Network
import utils
import os
import mnist_io

if __name__ == '__main__':
    dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "Python\\datasets")

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

    hidden_nodes = 150
    network = JIT_Network(input_shape=784, output_shape=10, node_count=784+hidden_nodes+10, learning_rate=0.005, id_num=0)
    
    for i in range(784):
        for j in range(784, 784+hidden_nodes):
            network.add_connection(j, i)

    for i in range(784, 784+hidden_nodes):
        for j in range(784+hidden_nodes, 784+hidden_nodes+10):
            network.add_connection(j, i)

    network.train(train_images, utils.jit_to_categorical(train_labels, 10), 1, 2000, 0.01)

    print(network.validate(test_images, utils.jit_to_categorical(test_labels, 10), utils.jit_categorical_compare))