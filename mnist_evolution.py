from evolutionary_module import evolve_node_count
import mnist_io
import datetime
import utils
import os

dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "Python\\datasets")

train_images = mnist_io.images_from_file(os.path.join(dataset, "train-images-idx3-ubyte\\train-images.idx3-ubyte"), 60000)
train_images = train_images.reshape(60000, 784).astype('float32')
train_images /= 255

train_labels = utils.jit_to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "train-labels-idx1-ubyte\\train-labels.idx1-ubyte"), 60000), 10)

test_images = mnist_io.images_from_file(os.path.join(dataset, "t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"), 10000)
test_images = test_images.reshape(10000, 784).astype('float32')
test_images /= 255

test_labels = utils.jit_to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), 10000), 10)

d = datetime.datetime.now()
with utils.OutSplit('mnist_evolution_{year}.{month}.{day}_{hour}.{minute}.{second}'.format(year=d.year, month=d.month, day=d.day, hour=d.hour, minute=d.minute, second=d.second)):
    evolved_network = evolve_node_count(train_images, train_labels, test_images, test_labels, utils.jit_categorical_compare, population_count=10, population_size=15, node_cap=1000, generations=50, target_accuracy=0.9, r=42)
    print(evolved_network.connections, "\n", evolved_network.weights, "\n",evolved_network.learning_rate)