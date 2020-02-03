import evolutionary_module as evm
import numpy as np
import mnist_io
import utils

r = 23
np.random.seed(r)

with utils.OutSplit('multiple_mnist'):
    x = mnist_io.images_from_file('datasets\\train-images-idx3-ubyte\\train-images.idx3-ubyte').reshape(60000, 784).astype('float64')/255
    y = utils.jit_to_categorical(mnist_io.labels_from_file('datasets\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte'), 10)
    val_x = mnist_io.images_from_file('datasets\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte').reshape(10000, 784).astype('float64')/255
    val_y = utils.jit_to_categorical(mnist_io.labels_from_file('datasets\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'), 10)

    # Grab input from the user and print it to stdout
    population_count = int(input('population_count: '))
    print(population_count)
    population_size = int(input('poplation_size: '))
    print(population_size)
    node_cap = int(input('node_cap: '))
    print(node_cap)
    generations = int(input('generations: '))
    print(generations)
    target_accuracy = float(input('target_accuracy: '))
    print('{}%'.format(target_accuracy*100))

    for i in range(20):
        best = evm.evolve_node_count(x, y, val_x, val_y, utils.jit_categorical_compare, population_count, population_size, node_cap, generations, target_accuracy, i*r+r)
        print("Best Network Accuracy:", best.validate(val_x, val_y, utils.jit_categorical_compare)*100, "%")