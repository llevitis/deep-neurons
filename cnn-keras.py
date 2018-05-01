import os, h5py
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from skimage import filters as skfilt
import matplotlib.pylab as plt
import models

batch_size = 10
num_classes = 3
epochs = 200

# input image dimensions
img_x, img_y = 128, 128

f = h5py.File(os.getcwd() + '/deep-neurons.hdf5', 'r')
labels = f['labels']
images = f['images']
images = np.reshape(images, (len(images),128,128,1))
for image in images:
    thresh_li = skfilt.threshold_li(image)
    mask = image < thresh_li
    image[mask] = 255

sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
train_indices, validation_indices, test_indices = None, None, None
for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
  train_indices = train_index
  validation_indices = validation_index

x_train = np.asarray(images)[train_indices]
y_train = np.asarray(labels)[train_indices]

x_validation = np.asarray(images)[validation_indices]
y_validation = np.asarray(labels)[validation_indices]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_validation.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                 activation='relu',
#                 input_shape=(128, 128, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))

#model.compile(loss=keras.losses.categorical_crossentropy,
 #             optimizer=keras.optimizers.Adam(),
#              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model = models.cnn()

model.fit(x_train, y_train,
         batch_size=batch_size,
          epochs=epochs,
         verbose=1,
          validation_data=(x_validation, y_validation),
          callbacks=[history])
score = model.evaluate(x_validation, y_validation, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

