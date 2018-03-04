import tensorflow as tf
import numpy as np
import math
from common import *


class Model:
    def __init__(self, output_dir, input, label, drop_out_keep, batch_size, summaries=True):
        self.devicePrior = '/gpu:0' if has_gpu() else '/cpu:0'

        self.model_save_dir = output_dir
        self.model_save_name = 'model.ckpt'
        self.summaries = summaries

        self.input_shape = input.shape
        self.label_shape = label.shape
        self.batch_size = batch_size
        self.dropOutKeep = drop_out_keep
        self.model_params = []

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.logits = self._inference(input)
        self.loss = self._loss(self.logits, label)
        self.train_op = self._train(self.loss)
        self.evaluation, self.evaluation_summary, self.evaluate_op = self._evaluate(self.logits, label)


        self.saver = tf.train.Saver(self.model_params, max_to_keep=2)

    def _inference(self, inputs):
        with tf.device(self.devicePrior):
            with tf.name_scope('fcc' ):
                d1 = self._make_dense(inputs, units=1500, activation_fn=tf.nn.relu, name="layer_1")
                d1 = self._make_dense(d1, units=1500, activation_fn=tf.nn.relu, name="layer_2")
                d1 = self._make_dense(d1, units=1500, activation_fn=tf.nn.relu, name="layer_3")
                d1 = self._make_dense(d1, units=1500, activation_fn=tf.nn.relu, name="layer_4")
                logits = self._make_dense(d1, units=self.label_shape[1].value, activation_fn=None, name="layer_out")
                return logits

    def _make_dense(self, ip_tensor, units, activation_fn, name):
        shape = [ip_tensor.shape[1].value, units]
        with tf.device(self.devicePrior):
            weights = self.get_weights(shape, name)
            biases = self.get_biases(shape, name)
            if activation_fn:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases)
                activation = activation_fn(activation, name=name + "_activation")
                activation = tf.nn.dropout(activation, keep_prob=self.dropOutKeep)
            else:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases, name=name + "_activation")

        if self.summaries:
            tf.summary.histogram(name + "_weights", weights)
            tf.summary.histogram(name + "_biases", biases)
            tf.summary.histogram(name + "_act", activation)
        self.model_params.append(weights)
        self.model_params.append(biases)
        return activation

    def _loss(self, logits, labels):
        with tf.device(self.devicePrior):
            with tf.name_scope('loss'):
                cast_labels = tf.cast(labels, dtype=tf.float32)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=cast_labels))
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=cast_labels))
                tf.summary.scalar('loss', loss)
                return loss

    def _train(self, loss):
        with tf.device(self.devicePrior):
            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(loss, global_step=self.global_step)
                return train_op

    def _evaluate(self, logits, labels):
        with tf.name_scope('evaluate'):
            precision, precision_op = tf.metrics.average_precision_at_k(labels, logits, k=2)
            summary_op = tf.summary.scalar('precision@2', precision)
            return precision, summary_op, precision_op


    def save(self, sess):
        self.saver.save(sess, self.model_save_dir + self.model_save_name, global_step=self.global_step)

    def restore(self, sess):
        path = tf.train.latest_checkpoint(self.model_save_dir)
        self.saver.restore(sess, path)

    def get_weights(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform(shape, minval=(-std), maxval=std), name=(name + "_weights"))

    def get_biases(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform([shape[-1]], minval=(-std), maxval=std), name=(name + "_biases"))