import tensorflow as tf
import numpy as np



x1 = tf.placeholder(tf.float32, shape=[10,5], name="x1")
x2 = tf.placeholder(tf.float32, shape=[10,5], name="x2")
labels = tf.placeholder(tf.float32, shape=[10,], name="label")

l2distance = tf.reduce_sum(tf.square(tf.subtract(x1,x2)), axis=1)
sqrt_l2 = tf.sqrt(l2distance)
squared_difference = tf.subtract(sqrt_l2, labels)
loss = tf.reduce_mean(squared_difference)
evaluation = tf.metrics.mean(squared_difference)

print("x1", x1)
print("x2", x2)
print("l2distance", l2distance)
print("squared_difference", squared_difference)
print("loss", loss)
print("evaluation", evaluation)

x1_ip = np.array([[20,  7, 16, 24,  7],
       [11,  8, 15, 10, 16],
       [19,  0, 24, 21, 21],
       [12,  7,  1,  3,  0],
       [ 7, 22,  5, 15,  0],
       [ 9, 11, 13, 23, 14],
       [19, 21, 23,  0, 12],
       [ 7, 23, 11, 15,  9],
       [ 8, 21, 15, 13, 24],
       [24,  7, 23, 18, 24]])

x2_ip = np.array([[ 5, 23, 14, 10, 24],
       [20, 18,  0, 13, 21],
       [15,  0, 14,  1, 21],
       [ 0,  8, 22,  9,  9],
       [15,  2, 23, 17, 11],
       [ 4, 23, 16, 11,  6],
       [ 5, 18,  2, 16,  6],
       [14,  3, 22,  0, 21],
       [12, 13, 18, 22, 13],
       [12,  0,  4,  3, 17]])

label_ip = np.array([17,  5, 10, 11,  7, 19, 12, 23, 13, 12])

with tf.Session() as sess:
    l2, sq, loss = sess.run([l2distance, squared_difference, loss], feed_dict={x1: x1_ip, x2: x2_ip, labels: label_ip})
    print("l2", l2)
    print("sq", sq)
    print("loss", loss)