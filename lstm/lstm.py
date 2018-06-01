import tensorflow as tf
import numpy as np
from tensorflow.contrib.training import HParams

from framework.network import Network
from framework.config import Config
from framework.serializer import Serializer
from autoencoder.encoder_writer import Kmer_Utility as ku


class LSTM_Classification(Network):
    def __init__(self, config):
        Network.__init__(self, config)
        self.model_save_filename = "lstm_mlp.ckpt"
        self.learning_rate = config.training.get('learning_rate', 0.001)

        lstm = config.model.get('lstm', None)
        mlp = config.model.get('multilayer', None)

        self.lstm_hparam = HParams(name=lstm.get('name', 'lstm'),
                                   hidden_size=lstm.get('hidden_units', -1),
                                   num_layers=lstm.get('num_layers', -1),
                                   dropout=1-lstm.get('dropout_keep', 0.7),
                                   bidirectional=lstm.get('type', 'unidirectional') == 'bidirectional'
                                   )
        self.mlp_hparam = HParams(name=mlp.get('name', 'mlp'),
                                  layers=mlp.get('layer', []),
                                  activations=mlp.get('activation', []),
                                  dropout_keep=mlp.get('dropout_keep', 0.7))
        self.mlp_hparam.layers.append(config['output_classes'])
        self.mlp_hparam.activations.append('None')

        self.train_op = None
        self.eval_mean_op = None
        self.eval_update_op = None
        self.prediction_op = None

    def _lstm_layer(self, input, params):
        mode = "cpu" if "cpu" in self.device else "gpu"
        with tf.device(self.device), tf.variable_scope(params.name, reuse=tf.AUTO_REUSE):
            if mode == "cpu":
                def make_cell(hidden_size, num_layers, dropout):
                    cell = tf.contrib.rnn.LSTMBlockCell
                    # cell = tf.nn.rnn_cell.BasicLSTMCell
                    cells_fw = [cell(hidden_size) for _ in range(num_layers)]
                    if dropout > 0.0:
                        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
                    return cells_fw

                cells_fw = make_cell(params.hidden_size, params.num_layers, params.dropout)
                if not params.bidirectional:
                    cells_fw = tf.contrib.rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
                    outputs, _ = tf.nn.dynamic_rnn(cell=cells_fw,
                                                   inputs=input,
                                                   dtype=tf.float32)
                else:
                    cells_bw = make_cell(params.hidden_size, params.num_layers, params.dropout)
                    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cells_fw=cells_fw,
                        cells_bw=cells_bw,
                        inputs=input,
                        dtype=tf.float32)
                return outputs
            elif mode == "gpu":
                t_input = tf.transpose(input, [1, 0, 2])
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=params.num_layers,
                    num_units=params.hidden_size,
                    dropout=params.dropout,
                    direction="bidirectional" if params.bidirectional else "unidirectional")
                outputs, _ = lstm(t_input)
                # Convert back from time-major outputs to batch-major outputs.
                outputs = tf.transpose(outputs, [1, 0, 2])
                return outputs

    def _mlp_layer(self, lstm_out, params, histogram=True):
        with tf.device(self.device), tf.variable_scope(params.name, reuse=tf.AUTO_REUSE):
            layer = tf.contrib.layers.flatten(lstm_out)
            for idx, (hidden, activation) in enumerate(zip(params.layers, params.activations)):
                layer = self._make_dense(layer, \
                                         units=hidden, \
                                         activation_fn=self._get_activation(activation), \
                                         dropout_keep=params.dropout_keep, \
                                         name="layer_"+str(idx) if idx!=len(params.layers)-1 else "out", \
                                         histogram=histogram)
            return layer


    def _inference(self, inputs, histogram=True):
        lstm_layer = self._lstm_layer(inputs, self.lstm_hparam)
        mlp_layer = self._mlp_layer(lstm_layer, self.mlp_hparam, histogram=histogram)
        return mlp_layer

    def train(self, X, Y):
        with tf.name_scope("train"):
            if self.train_op is None:
                logits = self._inference(X)
                self.loss_op = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
                self.model_scalars.append(tf.summary.scalar("loss", self.loss_op))
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                    .minimize(self.loss_op, global_step=tf.train.get_global_step())
            return self.train_op, self.loss_op


    def evaluation(self, X, Y):
        with tf.name_scope("evaluation"):
            if self.eval_update_op is None or self.eval_mean_op is None:
                self.eval_output_logit = self._inference(X)
                self.eval_mean_op, self.eval_update_op = tf.metrics.mean_per_class_accuracy(labels=tf.to_int64(Y), predictions=tf.round(tf.sigmoid(self.eval_output_logit)), num_classes=2)
                self.model_scalars.append(tf.summary.scalar("evaluation", self.eval_mean_op))
            return self.eval_mean_op, self.eval_update_op

    def prediction(self, X):
        with tf.name_scope("prediction"):
            if self.prediction_op is None:
                logits = self._inference(X, histogram=False)
                self.prediction_op = tf.sigmoid(logits)
            return self.prediction_op

    def summary_scalars(self):
        return tf.summary.merge(self.model_scalars)

    def summary_histograms(self):
        return tf.summary.merge(self.model_histogram)

    def _get_saver(self):
        return tf.train.Saver(max_to_keep=2)

    def getModelSaveFilename(self):
        return self.model_save_filename



k = 4
def time_step_vector(sequence):
    totKmers = len(sequence)-k+1
    toRet = []
    for i in range(totKmers):
        kmer = sequence[i:k+i]
        X = np.zeros((4**k))
        X[ku.encodeKmer(kmer)] = 1
        toRet.append(X)
    return np.asarray(toRet)

def make_dataset(sequences):
    dataset = tf.data.Dataset.from_generator(sequences).map(time_step_vector, num_parallel_calls=3)\
        .shuffle().repeat().batch(2).make_one_shot_iterator()
    return dataset.get_next()

def lstm(input, params, mode="cpu"):
    if mode == "cpu":
        cell = tf.contrib.rnn.LSTMBlockCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=input,
            dtype=tf.float32)
        return outputs
    elif mode == "gpu":
        t_input = tf.transpose(input, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=params.num_layers,
            num_units=params.num_nodes,
            dropout=params.dropout,
            direction="bidirectional")
        outputs, _ = lstm(t_input)
        # Convert back from time-major outputs to batch-major outputs.
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs


def test_serializer(sequences):
    writer = tf.python_io.TFRecordWriter("sample_lstm_run/serialization_test.tfrecords")
    serializer = Serializer({'x': 'sparse', 'y': 'sparse'})
    category = 1
    cat_vec = np.zeros(10)
    cat_vec[category] = 1
    batch = []
    for sequence in sequences:
        length = len(sequence)
        vector = time_step_vector(sequence)
        batch.append(vector)
        print("vector shape", vector.shape)
        vector_new = vector.reshape((-1))
        print("vector shape", vector_new.shape)
        serializable_string = serializer.make_serializable(x=vector_new, y=cat_vec)
        writer.write(serializable_string)
    writer.close()
    serializer.save_meta("sample_lstm_run/serialization_test.npy")

    # LSTM
    # CudnnLSTM is time - major
    placeholder = tf.placeholder(tf.float32, shape=(None, length-k+1, 4**k))
    hparams = HParams(num_nodes=10, num_layers=1, dropout=0.0)
    outputs = lstm(placeholder, params=hparams)
    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.layers.dense(outputs, 10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outs = sess.run(outputs, feed_dict={placeholder: batch})
        print(outs)



    def _separate(dictionary):
        X,Y = dictionary['x'], dictionary['y']
        X,Y = tf.cast(X, tf.float32), tf.cast(Y, tf.float32)
        X = tf.reshape(X, shape=(length-k+1, 4**k))
        return X,Y

    deserializer = Serializer("sample_lstm_run/serialization_test.npy")
    X,Y = tf.data.TFRecordDataset("sample_lstm_run/serialization_test.tfrecords").map(deserializer.deserialize)\
        .map(_separate) \
        .make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        try:
            while True:
                x,y = sess.run([X,Y])
                print("x.shape", x.shape)
                print("sum", np.sum(vector-x))
        except tf.errors.OutOfRangeError as ex:
            pass










def main():
    # sequences = ["GATATTCTTACGTGTAACGTAGCTATGTATTTTACAGAGCTGGCGTACGCGTTGAACACTTCACAGATGATAGGGATTCGGGTAAAGAGCGTGTTATTGGGGACTTACACAGGCGTAGACTACAATGGGCCCAACTCAATCACAGCTCGAGCGCCTTGAATAACGTACTCATCTCTATACATTCTCGACAATCTATCGAG",
    #             "TCGTCGCTGACGTTTACACTCTAGTCTCATTATAATCGTTCGCTATTCAGGGATTGACCAACACCGGAAAACATCTCACTTGAAGTAATGTATACGACAGAGTCCGTGCACCTACCAAACCTCTTTAGTCTAAGTTCAGACTAGTTGGAAGTTTGTCTAGATCTCAGATTTTGTCACTAGAGGACGCACGCTCTATTTTT",
    #             "ATGATCCATTGATGTCCCTGACGCTGCAAAATTTGCAACCAGGCAGTCTTCGCGGTAGGTCCTAGTGCAATGGGGCTTTTTTTCCATAGTCCTCGAGAGGAGGAGACGTCAGTCCAGATATCTTTGATGTCGTGATTGGAAGGACCCTTGGCCCTCCACCCTTAGGCAGTGTATACTCTTCCATAAACGGGCTATTAGTT"
    #             ]
    sequences = ["GTAGCTATGTATTTTACAGAGCTGGCGTACGCGTTGAACACTTCACAGATGATAGGGATTCGGGTAAAGAGCGTGTTATTGGGGACTTACACAGGCGTAGACTACAATGGGCCCAACTCAATCACAGCTCGAGCGCCTTGAATAACGTACTCATCTCTATACATTCTCGACAATCTATCGAGCGACTCGATTATCAACGGGTGTCTTGCAGTTCTAATCTCTTGCCAGCATCGTAATAGCCTCCAAGAGATTGATGATAGTCATGGGCACAGAGCTGAGACGGCGCCGATGGATAGCGGACTTTCGGTCAACCACAATTCCCCACGAGACAGGTCCTGCCGTGCGCATCACTCTGAATGTACAAGCAACCCAAGAGGGCTGAGCCTGGACTCAGCTGGTTCCTGGGTGAGCTCGAGACTCGGGGTGACAGCTCTTCATACATAGAGCGGGGGCGTCGAACGGTCGTGAAAGTCATAGTACCCCGGGTACCAACTTACTGA"]
    test_serializer(sequences)
    exit(1)
    config = Config("framework/sample_hparams/hparam-lstm.yaml")
    dataset = make_dataset(sequences)
    with tf.Session() as sess:
        batch = sess.run([dataset])
        print(batch)


if __name__ == "__main__":
    main()