#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:43:26 2021

@author: jialichen
"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.datasets import mnist


import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import CSVLogger
csv_logger = CSVLogger('6.1_log.txt', append=True, separator=';')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reference: https://github.com/ardendertat/Applied-Deep-Learning-with-Keras/blob/master/notebooks/Part%203%20-%20Autoencoders.ipynb

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)


input_size = 784
bottle_neck = 90
code_size = 32

input_img = Input(shape=(input_size,))
hidden_1 = Dense(bottle_neck, activation='relu')(input_img)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(bottle_neck, activation='relu')(code)
output_img = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=3)


# train the autoencoder
history = autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=20,
                callbacks = [csv_logger])
history_dict = history.history

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()
plt.savefig("6.1plot.png")


# apply to fashion data 
from tensorflow.keras.datasets import fashion_mnist

(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
x_train_fashion = x_train_fashion/np.max(x_train_fashion)
x_train_fashion = x_train_fashion.reshape((60000, 28*28))

x_train_fashion = x_train_fashion/255

threshold = 4 * autoencoder.evaluate(x_train,x_train)

result = autoencoder.predict(x_train_fashion)

count = 0  #count of anomaly 
for i in range(x_train_fashion.shape[0]):
    if np.mean((x_train_fashion[i] - result[i])**2) > threshold:
        count += 1
        
print('Total anomaly; ',count)
