import tensorflow as tf
import numpy as np
import _thread

from mnist_view import ViewData
import mnist_io

keras = tf.keras

from keras.models import Sequential
from keras.layers import Input, Dense

model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

train_images = mnist_io.images_from_file("datasets/train-images-idx3-ubyte/train-images.idx3-ubyte")
train_images = train_images.reshape(60000, 784).astype('float32')
train_images /= 255

train_labels = mnist_io.labels_from_file("datasets/train-labels-idx1-ubyte/train-labels.idx1-ubyte")

test_images = mnist_io.images_from_file("datasets/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte")

test_labels = mnist_io.labels_from_file("datasets/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte")

view = ViewData(test_images, test_labels)

test_images = test_images.reshape(10000, 784).astype('float32')
test_images /= 255

history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels))

output = model.predict(test_images[:10])
for case in output:
    m=max(case)
    print([i for i, j in enumerate(case) if j == m])
