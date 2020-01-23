import tensorflow as tf
import numpy as np
import _thread
import os

from mnist_view import ViewData
import mnist_io

keras = tf.keras

from keras.models import Sequential
from keras.layers import Input, Dense

dataset = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "Python\\datasets")

train_images = mnist_io.images_from_file(os.path.join(dataset, "train-images-idx3-ubyte\\train-images.idx3-ubyte"), 60000)
train_images = train_images.reshape(60000, 784).astype('float32')
train_images /= 255

train_labels = keras.utils.to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "train-labels-idx1-ubyte\\train-labels.idx1-ubyte"), 60000), 10)

test_images = mnist_io.images_from_file(os.path.join(dataset, "t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte"), 10000)
test_images = test_images.reshape(10000, 784).astype('float32')
test_images /= 255

test_labels = keras.utils.to_categorical(mnist_io.labels_from_file(os.path.join(dataset, "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"), 10000), 10)

# view = ViewData(test_images, test_labels)

model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels))

""" 
output = model.predict(test_images[:10])
for case in output:
    m=max(case)
    print([i for i, j in enumerate(case) if j == m])
"""
