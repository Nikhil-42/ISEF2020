import tensorflow as tf
import numpy as np
from viewimages import ViewImages

keras = tf.keras

from keras.models import Sequential
from keras.layers import Input, Dense

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=1000))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

view = ViewImages("datasets/train-images-idx3-ubyte/train-images.idx3-ubyte")