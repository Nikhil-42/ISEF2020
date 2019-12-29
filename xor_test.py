import tensorflow as tf
import numpy as np
import _thread

from mnist_view import ViewData
import mnist_io

keras = tf.keras

from keras.models import Sequential
from keras.layers import Input, Dense

model = Sequential()

model.add(Dense(units=2, activation='relu', input_shape=(2,)))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd',metrics=['accuracy'], loss='sparse_categorical_crossentropy')

x = [(1, 0) if num < 0.25 else (0, 1) if num < 0.5 else (1, 1) if num < 0.75 else (0,0) for num in np.random.rand(100)]
y = [(1,) if sum(itm)==0 else (0,) for itm in x]

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

history = model.fit(x, y, validation_data=(x, y), epochs=100)

print(model.predict(np.array(((1, 0),(0, 0)))))

