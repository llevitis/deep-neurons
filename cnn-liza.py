# import the necessary packages
import h5py
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import tensorflow as tf
import keras
from skimage import filters as skfilt

sess = tf.InteractiveSession()
f = h5py.File(os.getcwd() + '/deep-neurons.hdf5', 'r')
labels = f['labels']
images = f['images']
BATCH_SIZE = 20

sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
train_indices, validation_indices, test_indices = None, None, None
for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
  train_indices = train_index
  validation_indices = validation_index

images_train = np.asarray(images)[train_indices]
labels_train = np.asarray(labels)[train_indices]

images_validation = np.asarray(images)[validation_indices]
labels_validation = np.asarray(labels)[validation_indices]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 128*128])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

# FIRST CONVOLUTIONAL LAYER
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 128, 128, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# reduce image size to 64*64
h_pool1 = max_pool_2x2(h_conv1)

# SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# reduce image size to 32*32
h_pool2 = max_pool_2x2(h_conv2)

# DENSELY CONNECTED LAYER
W_fc1 = weight_variable([32 * 32 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DROPOUT (to reduce overfitting - turn dropout on during training & off during testing)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# TRAIN & EVALUATE
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# use ADAM optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
for epoch in range(5):
    print("Training epoch: " + str(epoch))
    for index in range(int(images_train.shape[0] / BATCH_SIZE)):
        images_batch = np.reshape(images_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE], (BATCH_SIZE, 128*128))
        #import pdb; pdb.set_trace()
        labels_batch = labels_train[index * BATCH_SIZE:(index+1) * BATCH_SIZE]
        train_accuracy = accuracy.eval(feed_dict={
            x: images_batch, y_: labels_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (epoch, train_accuracy))
        train_step.run(feed_dict={x: images_batch, y_: labels_batch, keep_prob: 0.5})

images_validation = np.reshape(images_validation, (len(images_validation), 128*128))
#import pdb; pdb.set_trace()
#print(accuracy.eval(session=sess, feed_dict={x: images_validation, y_: labels_validation}))
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: images_validation, y_: labels_validation, keep_prob: 1.0}))
