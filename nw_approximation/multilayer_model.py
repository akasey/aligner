import math
import numpy as np
import tensorflow as tf


class MultiLayerModel():
    def __init__(self, config):
        self.device = config.training.get('device', '/cpu:0')
        self.model_save_location = config.runtime.get('model_dir', '.') + "/tensorboard"
        self.model_save_filename = "/multilayermodel.ckpt"
        self.learning_rate = config.training.get('learning_rate', 0.0001)

        self.name = config.model['multilayer'].get('name', 'fcc')
        self.dropout_keep = config.model['multilayer'].get('dropout_keep', 0.7)
        self.hidden_layers = config.model['multilayer']['layer']
        self.activations = config.model['multilayer']['activation']
        self.input_shape = [None, config.input_features]
        self.out_shape = [None, self.hidden_layers[-1]]

        self.model_matrices = {}
        self.model_histogram = []
        self.model_scalars = []

        self.train_op = None
        self.eval_mean_op = None
        self.eval_update_op = None
        self.prediction_op = None
        self.saver = None


    def _get_activation(self, str):
        return eval(str)

    def _get_weights(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.get_variable(name+"_weights", shape, initializer=tf.truncated_normal_initializer(stddev=std))

    def _get_biases(self, shape, name):
        return tf.get_variable(name + "_biases", [shape[-1]], initializer=tf.constant_initializer(value=0.01))

    def _make_dense(self, ip_tensor, units, activation_fn, dropout_keep, name):
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

        self.model_histogram.append(tf.summary.histogram(name + "_weights", weights))
        self.model_histogram.append(tf.summary.histogram(name + "_biases", biases))
        self.model_histogram.append(tf.summary.histogram(name + "_act", activation))

        if not weights.name in self.model_matrices:
            self.model_matrices[weights.name] = weights
        if not biases.name in self.model_matrices:
            self.model_matrices[biases.name] = biases
        return activation

    def _encoder(self, inputs, dropout_keep):
        with tf.device(self.device), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            layer = inputs
            for idx, (hidden, activation) in enumerate(zip(self.hidden_layers, self.activations)):
                layer = self._make_dense(layer, \
                                         units=hidden, \
                                         activation_fn=self._get_activation(activation), \
                                         dropout_keep=dropout_keep, \
                                         name="layer_"+str(idx) if idx!=len(self.hidden_layers)-1 else "out")
            return layer

    """
    Very specific for the nw_approximation
    """
    def train(self, X, Y):
        if self.train_op == None:
            with tf.device(self.device), tf.name_scope('train'):
                x1 = self._encoder(X['x1'], self.dropout_keep)
                x2 = self._encoder(X['x2'], self.dropout_keep)
                l2distance = tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)
                sqrt_l2 = tf.sqrt(l2distance)
                squared_difference = tf.square(tf.subtract(sqrt_l2, Y))
                self.loss_op = tf.reduce_mean(squared_difference)
                self.model_scalars.append(tf.summary.scalar("loss", self.loss_op))

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = optimizer.minimize(self.loss_op, global_step=tf.train.get_global_step())
        return self.train_op, self.loss_op

    def evaluation(self, X, Y):
        if self.eval_mean_op == None or self.eval_update_op == None:
            with tf.device(self.device), tf.name_scope('evaluation'):
                x1 = self._encoder(X['x1'], dropout_keep=1.0)
                x2 = self._encoder(X['x2'], dropout_keep=1.0)
                l2distance = tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)
                sqrt_l2 = tf.sqrt(l2distance)
                squared_difference = tf.square(tf.subtract(sqrt_l2, Y))
                self.eval_mean_op, self.eval_update_op = tf.metrics.mean(squared_difference)
                self.model_scalars.append(tf.summary.scalar("evaluation", self.eval_mean_op))
        return self.eval_mean_op, self.eval_update_op

    def prediction(self, X):
        if self.prediction_op == None:
            with tf.device(self.device), tf.name_scope('prediction'):
                self.prediction_op = self._encoder(X['x1'], dropout_keep=1.0)
        return self.prediction_op

    def summary_scalars(self):
        return tf.summary.merge(self.model_scalars)

    def summary_histograms(self):
        return tf.summary.merge(self.model_histogram)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.out_shape

    def save(self, sess):
        if not self.saver:
            self.saver = tf.train.Saver(self.model_matrices, max_to_keep=2)
        self.saver.save(sess, self.model_save_location + self.model_save_filename, global_step=tf.train.get_global_step())

    def restore(self, sess):
        if not self.saver:
            self.saver = tf.train.Saver(self.model_matrices, max_to_keep=2)
        path = tf.train.latest_checkpoint(self.model_save_location)
        if path:
            self.saver.restore(sess, path)
            return True
        else:
            return False



