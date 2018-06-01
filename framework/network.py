import numpy as np
import math
import tensorflow as tf
import os

class Network():
    def __init__(self, config):
        self.config = config
        self.device = config.training.get('device', '/cpu:0')
        self.model_save_location = config.runtime.get('model_dir', '.') + "/tensorboard"

        self.model_matrices = {}
        self.model_histogram = []
        self.model_scalars = []

        self.saver = None

    def _get_activation(self, str):
        return eval(str)

    def _get_weights(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.get_variable(name+"_weights", shape, initializer=tf.truncated_normal_initializer(stddev=std))

    def _get_biases(self, shape, name):
        return tf.get_variable(name + "_biases", [shape[-1]], initializer=tf.constant_initializer(value=0.0001))

    def _make_dense(self, ip_tensor, units, activation_fn, dropout_keep, name, histogram=True):
        shape = [ip_tensor.shape[1].value, units]
        with tf.device(self.device):
            weights = self._get_weights(shape, name)
            biases = self._get_biases(shape, name)
            if activation_fn:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases)
                activation = activation_fn(activation, name=name + "_activation")
                activation = tf.nn.dropout(activation, keep_prob=dropout_keep)
            else:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases, name=name + "_activation")

        if histogram:
            self.model_histogram.append(tf.summary.histogram(name + "_weights", weights))
            self.model_histogram.append(tf.summary.histogram(name + "_biases", biases))
            self.model_histogram.append(tf.summary.histogram(name + "_act", activation))

        if not weights.name in self.model_matrices:
            self.model_matrices[weights.name] = weights
        if not biases.name in self.model_matrices:
            self.model_matrices[biases.name] = biases
        return activation

    def summary_scalars(self):
        return tf.summary.merge(self.model_scalars)

    def summary_histograms(self):
        return tf.summary.merge(self.model_histogram)

    def _get_saver(self):
        return tf.train.Saver(self.model_matrices, max_to_keep=2)

    def save(self, sess):
        if not self.saver:
            self.saver = self._get_saver()
        self.saver.save(sess, self.model_save_location + "/" + self.getModelSaveFilename(), global_step=tf.train.get_global_step())
        if not os.path.exists(self.model_save_location + "/model.pb"):
            tf.train.write_graph(sess.graph.as_graph_def(), self.model_save_location, "model.pb")

    def restore(self, sess):
        if not self.saver:
            self.saver = self._get_saver()
        path = tf.train.latest_checkpoint(self.model_save_location)
        if path:
            self.saver.restore(sess, path)
            return True
        else:
            return False

    def getModelSaveFilename(self):
        return "network-abstract.ckpt"