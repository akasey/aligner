import tensorflow as tf
import numpy as np
import argparse

from autoencoder.encoder_writer import Kmer_Utility as ku
from mlp.classification_loader import Classification_Loader, Classification_Writer, logger


class LSTMClassificationLoader(Classification_Loader):
    def __init__(self, dirname, batch_size):
        Classification_Loader.__init__(self, dirname, batch_size)
        self.x_shape = None

    def _get_shape(self):
        if self.x_shape is None:
            input_shape = self.meta['input_shape'].split('x')
            self.x_shape = (int(input_shape[0]), int(input_shape[1]))
        return self.x_shape

    def _separate(self, dictionary):
        X,Y = Classification_Loader._separate(self, dictionary)
        X = tf.reshape(X, shape=self._get_shape())
        return X,Y


class LSTMClassificationWriter(Classification_Writer):
    def __init__(self, dirname):
        Classification_Writer.__init__(self, dirname)
        self.strides = 1
        self.K = 4

        self.segment_length = 5000
        self.window_length = 200
        self.mutation_freq = 5
        self.mutation_prob = 0.2
        self.unknown_window_fraction = 0.2

    def _to_time_series(self, window):
        time_series = []
        for start in range(0, len(window), self.strides):
            if start+self.K < len(window):
                time_series.append(window[start:start+self.K])
        return time_series


    def _one_hot_input(self, window, reverse=False):
        window = ku.reverse_complement(window) if reverse else window
        # each kmer is a timestep in LSTM
        time_series = self._to_time_series(window)
        toRet = np.zeros((len(time_series), 4 ** self.K))
        rows = range(len(time_series))
        cols = [ku.encodeKmer(timestep) for timestep in time_series]
        toRet[rows,cols] = 1
        self._register_meta('input_shape', str(toRet.shape[0])+'x'+str(toRet.shape[1]))
        return toRet.reshape((-1))

    @logger
    def write_tf(self, df, allWindows, numSegments, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for window in df:
                input = self._one_hot_input(window, reverse=False)
                output = self._one_hot_output(allWindows[window], numSegments)
                self._register_meta('output_shape', output.shape[0])
                serializable_features = self.serialization.make_serializable(input=input, output=output)
                writer.write(serializable_features)
                # Reverse window
                input2 = self._one_hot_input(window, reverse=True)
                serializable_features = self.serialization.make_serializable(input=input2, output=output)
                writer.write(serializable_features)

    def write(self):
        self._register_meta('window_length', self.window_length)
        self._register_meta('K', self.K)
        Classification_Writer.write(self)


def main():
    if FLAGS.mode == "read":
        reader = LSTMClassificationLoader(FLAGS.data_dir, batch_size = 11)
        X,Y = reader.load_dataset("train")
        with tf.Session() as sess:
            _x, _y = sess.run([X,Y])
            print("x shape", _x.shape)
            print("y shape", _y.shape)
    elif FLAGS.mode == "write":
        writer = LSTMClassificationWriter(FLAGS.data_dir)
        writer.write()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        # default="sample_classification_run/",
        default="sample_lstm_run/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--mode",
        type=str,
        default="read",
        help="Modes: {write, read}"
    )

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()
