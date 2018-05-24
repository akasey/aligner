import argparse
import random

import numpy as np
import tensorflow as tf
from encoder_writer import Kmer_Utility as ku
from sklearn.model_selection import train_test_split

from framework.common import make_logger
from framework.serializer import Serializer


class DenoisingLoader:
    def __init__(self, dirname, batch_size):
        self.directory = dirname
        self.batch_size = batch_size

        self.meta = self._load_meta(self.directory)
        self.serialization = Serializer(self.directory + "/serialization-meta.npy")
        self.train_dataset = None
        self.test_dataset = None

    def _load_meta(self, directory):
        filename = directory + "/meta.npy"
        metaTemp = np.load(filename)
        meta = {}
        for k in metaTemp.item():
            meta[k] = metaTemp.item().get(k)
        return meta

    def _deserialize_file(self, filename, parallelism=4):
        return tf.data.TFRecordDataset(filename).map(self.serialization.deserialize, num_parallel_calls=parallelism)

    def __separate(self, dictionary):
        X,Y = dictionary['input'], dictionary['output']
        X,Y = tf.cast(X, tf.float32), tf.cast(Y, tf.float32)
        return X,Y

    def load_dataset(self, type):
        filename = "train.tfrecords" if type=="train" else "test.tfrecords"
        filename = self.directory + "/" + filename
        if type == "train":
            if self.train_dataset is None:
                self.train_dataset = self._deserialize_file(filename)
            dataset = self.train_dataset
        elif type == "test":
            if self.test_dataset is None:
                self.test_dataset = self._deserialize_file(filename)
            dataset = self.test_dataset
        dataset = dataset.map(self.__separate).shuffle(1000).batch(self.batch_size).repeat()
        return dataset.make_one_shot_iterator().get_next()

    @property
    def train_size(self):
        if self.meta:
            return self.meta['train_size']
        return -1

    @property
    def test_size(self):
        if self.meta:
            return self.meta['test_size']
        return -1


class DenoisingWriter():
    def __init__(self, dirname):
        self.logger = make_logger("DenoisingWriter")
        self.directory = dirname
        self.K = 8
        self.window_length = 1000
        self.strides = self.K-1
        self.mutation_freq = 3
        self.mutation_prob = 0.2

        self.nucArr = ['A', 'C', 'G', 'T']
        self.meta = {}

    def __register_meta(self, key, value):
        if key not in self.meta:
            self.meta[key] = value

    def _read_windows_segments(self, fasta):
        allWindows = {}
        genome = ku.readGenome(fasta)
        wins = ku.slidingWindow(segment=genome, winlength=self.window_length, strides=self.strides)
        for win in wins:
            allWindows[win] = win
        return allWindows

    def __mutate(self, sequence, probability):
        seq = list(sequence)
        rand = np.random.rand(len(sequence))
        mutIdx = np.argwhere(rand <= probability).flatten()
        for i in mutIdx:
            seq[i] = self.nucArr[random.randint(0, 3)]
        return "".join(seq)

    def _one_hot_input(self, window):
        bow = ku.encodeKmerBagOfWords(window, K=self.K)
        b = np.zeros(len(bow))
        b[np.argwhere(bow)] = 1
        return b

    def write_tf(self, df, allWindows, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for window in df:
                input = self._one_hot_input(window)
                output = self._one_hot_input(allWindows[window])
                serializable_features = self.serialization.make_serializable(input=input, output=output)
                writer.write(serializable_features)

    def _create_serializer(self):
        self.serialization = Serializer({'input': 'sparse', 'output': 'sparse'})

    def _test_train_split(self, allWindows):
        train, test = train_test_split([*allWindows], test_size=0.30)
        return train, test

    def _export_meta(self):
        np.save(self.directory +"/meta.npy", self.meta)
        self.serialization.save_meta(self.directory+"/serialization-meta.npy")

    def _introduce_mutations(self, allWindows):
        allMutations = {}
        for window, segIds in allWindows.items():
            for i in range(self.mutation_freq):
                mutated_window = self.__mutate(window, self.mutation_freq)
                allMutations[mutated_window] = window

        # update the allWindows dict with allMutations
        allWindows.update(allMutations)
        return allWindows

    def write(self):
        self.logger.info("Initiating...")
        allWindows = self._read_windows_segments(self.directory +"/sequence.fasta")
        allWindows = self._introduce_mutations(allWindows)
        train, test = self._test_train_split(allWindows)
        self.__register_meta('train_size', len(train))
        self.__register_meta('test_size', len(test))
        self.__register_meta('total', len(allWindows))
        self._create_serializer()
        self.write_tf(train, allWindows, self.directory+"/train.tfrecords")
        del train
        self.write_tf(test, allWindows, self.directory+"/test.tfrecords")
        del test, allWindows
        self._export_meta()


def write_main():
    writer = DenoisingWriter(FLAGS.data_dir)
    writer.write()

def read_main():
    loader = DenoisingLoader(FLAGS.data_dir, batch_size=1)
    features, labels = loader.load_dataset("test")
    sess = tf.Session()
    for i in range(2):
        fl, ll = sess.run([features, labels])
        print(fl, fl.shape)
        print(ll, ll.shape)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_denoising_run/",
        help="Where is input data dir? use data_generation.py to create one")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    # read_main()
    write_main()
