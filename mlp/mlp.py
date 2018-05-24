import tensorflow as tf

from framework.network import Network


class MultiLayerPerceptron(Network):
    def __init__(self, config):
        Network.__init__(self, config)
        self.model_save_filename = config.model['multilayer'].get('name', 'mlp-model') + ".ckpt"
        self.learning_rate = config.training.get('learning_rate', 0.001)

        self.name = config.model['multilayer'].get('name', 'mlp')
        self.dropout_keep = config.model['multilayer'].get('dropout_keep', 0.7)
        self.hidden_layers = config.model['multilayer']['layer']
        self.activations = config.model['multilayer']['activation']
        self.input_shape = [None, config.input_features]
        self.out_shape = [None, config['output_classes']]

        self.hidden_layers.append(config['output_classes'])
        self.activations.append('None')

        self.train_op = None
        self.eval_mean_op = None
        self.eval_update_op = None
        self.prediction_op = None

    def _inference(self, inputs, dropout_keep, histogram=True):
        with tf.device(self.device), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            layer = inputs
            for idx, (hidden, activation) in enumerate(zip(self.hidden_layers, self.activations)):
                layer = self._make_dense(layer, \
                                         units=hidden, \
                                         activation_fn=self._get_activation(activation), \
                                         dropout_keep=dropout_keep, \
                                         name="layer_"+str(idx) if idx!=len(self.hidden_layers)-1 else "out", \
                                         histogram=histogram)
            return layer

    def train(self, X, Y):
        with tf.name_scope("train"):
            if self.train_op is None:
                logits = self._inference(X, self.dropout_keep)
                self.loss_op = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
                self.model_scalars.append(tf.summary.scalar("loss", self.loss_op))
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                    .minimize(self.loss_op, global_step=tf.train.get_global_step())
            return self.train_op, self.loss_op

    def evaluation(self, X, Y):
        with tf.name_scope("evaluation"):
            if self.eval_update_op is None or self.eval_mean_op is None:
                self.eval_output_logit = self._inference(X, dropout_keep=1.0)
                # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
                # self.eval_mean_op, self.eval_update_op = tf.metrics.mean(cross_entropy)
                self.eval_mean_op, self.eval_update_op = tf.metrics.mean_per_class_accuracy(labels=tf.to_int64(Y), predictions=tf.round(tf.sigmoid(self.eval_output_logit)), num_classes=2)
                self.model_scalars.append(tf.summary.scalar("evaluation", self.eval_mean_op))
            return self.eval_mean_op, self.eval_update_op

    def prediction(self, X):
        with tf.name_scope("prediction"):
            if self.prediction_op is None:
                logits = self._inference(X, dropout_keep=1.0, histogram=False)
                self.prediction_op = tf.sigmoid(logits)
            return self.prediction_op

    def summary_scalars(self):
        return tf.summary.merge(self.model_scalars)

    def summary_histograms(self):
        return tf.summary.merge(self.model_histogram)

    # def get_input_shape(self):
    #     return self.input_shape
    #
    # def get_output_shape(self):
    #     return self.out_shape

    def getModelSaveFilename(self):
        return self.model_save_filename


