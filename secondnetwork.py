import neuralnetwork.network as net
import mnist_io
import mnist_view

def to_categorical(narry, classes: int) -> list:
    return [[0 if val!=i else 1 for i in range(classes)] for val in narry]

def from_categorical(narray) -> list:
    return [array.index(max(array)) for array in narray]

network = net.Network(input_shape=784, output_shape=10, node_count=784+512+10, activation_func=net.leaky_relu, learning_rate=0.000001)

network.add_connections(range(784, 784+512), range(0, 784))
network.add_connections(range(784+512, 784+512+10), range(784, 784+512))
#network.add_connections(range(784+512+512, 784+512+512+10), range(784+512, 784+512+512))

set_count = 30

train_images = mnist_io.images_from_file("datasets/train-images-idx3-ubyte/train-images.idx3-ubyte", set_count)
train_images = train_images.reshape(set_count, 784).astype('float32')
train_images /= 255

train_labels = mnist_io.labels_from_file("datasets/train-labels-idx1-ubyte/train-labels.idx1-ubyte", set_count)

test_images = mnist_io.images_from_file("datasets/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte", set_count)
test_images = test_images.reshape(set_count, 784).astype('float32')
test_images /= 255

test_labels = mnist_io.labels_from_file("datasets/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte", set_count)

# view = mnist_view.ViewData(train_images.reshape(set_count, 28, 28), train_labels)

network.train(train_images.tolist(), to_categorical(train_labels, 10), 2)

print(from_categorical(network.predict(test_images.tolist())))