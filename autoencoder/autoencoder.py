import tensorflow as tf
from framework.network import Network

class AutoEncoder(Network):
    def __init__(self, config):
        Network.__init__(self, config)
        self.layers = config.model['autoencoder']["layers"]
        self.encoding_layer = config.model['autoencoder']['encoding_layer']
        self.encoding_activation = config.model['autoencoder']['encoding_activation']
        self.dropout_keep = config.model['autoencoder'].get('dropout_keep', 0.7)
        self.activations = config.model['autoencoder']['activation']
        self.input_shape = [None, config.input_features]
        self.out_shape = self.input_shape
        self.name = config.model['autoencoder'].get('name', 'fcc')
        self.learning_rate = config.training.get('learning_rate', 0.001)

        self.loss_op = None
        self.train_op = None
        self.eval_mean_op = None
        self.eval_update_op = None
        self.prediction_op = None


    def _Wx_b(self, x, W, b, activation, name, dropout_keep=None):
        activation_fn = self._get_activation(activation)
        if activation_fn:
            activation_raw = tf.add(tf.matmul(x, W), b)
            activation = activation_fn(activation_raw, name=name + "_activation")
            if dropout_keep:
                activation = tf.nn.dropout(activation, keep_prob=dropout_keep)
        else:
            activation_raw = tf.add(tf.matmul(x,W), b, name= name +"_activation")
            activation = activation_raw

        self.model_histogram.append(tf.summary.histogram(name + "_weights", W))
        self.model_histogram.append(tf.summary.histogram(name + "_biases", b))
        self.model_histogram.append(tf.summary.histogram(name + "_act", activation))

        if not W.name in self.model_matrices and  "transpose" not in W.name:
            self.model_matrices[W.name] = W
        if not b.name in self.model_matrices:
            self.model_matrices[b.name] = b
        return activation_raw, activation

    def _encoder(self, inputs, dropout_keep):
        encoder_weights = []
        with tf.device(self.device), tf.name_scope("encoder"), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            ip_tensor = inputs
            for idx, (hidden, activation_fn) in enumerate(zip(self.layers, self.activations)):
                shape = [ip_tensor.shape[1].value, hidden]
                name = "layer_"+str(idx)
                W = self._get_weights(shape, name)
                b = self._get_biases(shape, name)
                encoder_weights.append(W)
                activation_raw, activation = self._Wx_b(x=ip_tensor, W=W, b=b, activation=activation_fn, name=name, dropout_keep=dropout_keep)
                ip_tensor = activation
            return ip_tensor, encoder_weights

    def _decoder(self, inputs, dropout_keep, rev_encoder_weights):
        with tf.device(self.device), tf.name_scope("decoder"), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            ip_tensor = inputs
            for idx, (hidden, activation_fn) in enumerate(zip(reversed(self.layers), reversed(self.activations))):
                name = "decoder_layer_"+str(idx)
                shape = [ip_tensor.shape[1].value, hidden]
                W = tf.transpose(rev_encoder_weights[idx])
                b = self._get_biases(shape, name)
                activation_raw, activation = self._Wx_b(x=ip_tensor, W=W, b=b, activation=activation_fn, name=name, dropout_keep=dropout_keep)
                ip_tensor = activation
            return ip_tensor

    def _projection(self, encoder_out, dropout_keep, encoder_weights):
        with tf.device(self.device), tf.name_scope("projection"), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            shape = [encoder_out.shape[1].value, self.encoding_layer]
            W = self._get_weights(shape, name="projection")
            b = self._get_biases(shape, name="projection")
            encoder_weights.append(W)
            activation_raw, activation = self._Wx_b(x=encoder_out, W=W, b=b, activation=self.encoding_activation, name="projection", dropout_keep=dropout_keep)
            return activation_raw, activation

    def _output(self, decoder_out, dropout_keep):
        with tf.device(self.device), tf.name_scope("output"), tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            shape = [decoder_out.shape[1].value, self.input_shape[1]]
            name = "output"
            W = self._get_weights(shape, name=name)
            b = self._get_biases(shape, name=name)
            activation_raw, activation = self._Wx_b(x=decoder_out, W=W, b=b, activation="None", name=name, dropout_keep=dropout_keep)
            return activation

    def _form_encode_decode_chain(self,X, dropout_keep):
        encoder_block, encoder_weights = self._encoder(X, dropout_keep)
        projection, decoder_in_activation = self._projection(encoder_block, dropout_keep, encoder_weights)
        encoder_weights.reverse()
        decoder_block = self._decoder(decoder_in_activation, dropout_keep, encoder_weights)
        output = self._output(decoder_block, dropout_keep)
        return output


    def train(self, X, Y):
        if self.train_op == None:
            with tf.device(self.device), tf.name_scope('train'):
                chain = self._form_encode_decode_chain(X, self.dropout_keep)
                logits = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=chain)
                self.loss_op = tf.reduce_mean(logits)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_op = optimizer.minimize(self.loss_op)
                self.model_scalars.append(tf.summary.scalar("loss", self.loss_op))
        return self.train_op, self.loss_op

    def evaluation(self, X, Y):
        if self.eval_mean_op == None or self.eval_update_op == None:
            with tf.device(self.device), tf.name_scope('evaluation'):
                chain = self._form_encode_decode_chain(X, 1.0)
                sigmoid_out = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=chain)
                self.eval_mean_op, self.eval_update_op = tf.metrics.mean(sigmoid_out)
                self.model_scalars.append(tf.summary.scalar("evaluation", self.eval_mean_op))
        return self.eval_mean_op, self.eval_update_op

    def prediction(self, X):
        if self.prediction_op == None:
            with tf.device(self.device), tf.name_scope('prediction'):
                encoder_block, encoder_weights = self._encoder(X, 1.0)
                projection, decoder_in_activation = self._projection(encoder_block, 1.0)
                self.prediction_op = projection
        return self.prediction_op

    def summary_scalars(self):
        return tf.summary.merge(self.model_scalars)

    def summary_histograms(self):
        return tf.summary.merge(self.model_histogram)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.out_shape

    def getModelSaveFilename(self):
        return "autoencoder.ckpt"