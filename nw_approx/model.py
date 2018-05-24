from framework.network import Network

import tensorflow as tf

class Encoder_Model(Network):
    def __init__(self, config):
        Network.__init__(self, config)
        self.model_save_filename = config.model['multilayer'].get('name', 'multilayermodel') + ".ckpt"
        self.learning_rate = config.training.get('learning_rate', 0.0001)

        self.name = config.model['multilayer'].get('name', 'fcc')
        self.dropout_keep = config.model['multilayer'].get('dropout_keep', 0.7)
        self.hidden_layers = config.model['multilayer']['layer']
        self.activations = config.model['multilayer']['activation']
        self.input_shape = [None, config.input_features]
        self.out_shape = [None, self.hidden_layers[-1]]

        self.train_op = None
        self.eval_mean_op = None
        self.eval_update_op = None
        self.prediction_op = None

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

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.out_shape

    def getModelSaveFilename(self):
        return self.model_save_filename



