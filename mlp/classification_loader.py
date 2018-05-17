import argparse
import random
import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import train_test_split

from autoencoder.encoder_writer import Kmer_Utility as ku
from framework.common import make_logger
from framework.serializer import Serializer
from fastamm import FastaMM

logging = make_logger("classification_loader.py")
def logger(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        logging.info('Running %s' % fn.__name__)
        out = fn(*args, **kwargs)
        logging.info('Completed running %s' % fn.__name__)
        return out

    return wrapper

class Classification_Loader:
    def __init__(self, dirname, batch_size):
        self.directory = dirname
        self.batch_size = batch_size
        self.meta = self._load_meta(self.directory)
        self.serialization = Serializer(self.directory+"/serialization-meta.npy")
        self.training_dataset = None
        self.test_dataset = None


    def _load_meta(self, directory):
        filename = directory + "/meta.npy"
        if os.path.exists(filename):
            metaTemp = np.load(filename)
            meta = {}
            for k in metaTemp.item():
                meta[k] = metaTemp.item().get(k)
            return meta
        return None

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
            if self.training_dataset is None:
                self.training_dataset = self._deserialize_file(filename)
            dataset = self.training_dataset
        elif type == "test":
            if self.test_dataset is None:
                self.test_dataset = self._deserialize_file(filename)
            dataset = self.test_dataset
        dataset = dataset.map(self.__separate).shuffle(1000).batch(self.batch_size).repeat()
        return dataset.make_one_shot_iterator().get_next()

    @property
    def train_size(self):
        return self.meta['train_size']

    @property
    def test_size(self):
        return self.meta['test_size']

class Classification_Writer:
    def __init__(self, dirname):
        self.logger = make_logger("ClassificationWriter")
        self.directory = dirname
        self.output_shards = 5
        self.K = 7
        self.segment_length = 5000
        self.window_length = 500
        self.strides = self.K-1
        self.mutation_freq = 3
        self.mutation_prob = 0.2
        self.unknown_window_fraction = 0.2

        self.nucArr = ['A', 'C', 'G', 'T']
        self.meta = {}

    @logger
    def _read_windows_segments(self, fastamm):
        allWindows = {}
        totalSegments = []
        for segID, segment in fastamm.allClassificationJob():
            totalSegments.append(segID)
            wins = ku.slidingWindow(segment=segment, winlength=self.window_length, strides=self.strides)
            for win in wins:
                if win in allWindows:
                    allWindows[win].append(segID)
                else:
                    allWindows[win] = [segID]

        numSegments = len(totalSegments)
        return numSegments, allWindows

    @logger
    def _add_unknown_windows(self, numSegments, allWindows):
        known_samples = []
        mutations = {}
        for window, segIds in allWindows.items():
            if random.random() <= self.unknown_window_fraction:
                known_samples.append(window)
        for sample in known_samples:
            while True:
                mutation_probability = min(self.mutation_prob + 0.1 + random.random(), 1.0)
                mutated_sequence = self.__mutate(sample, mutation_probability)
                if mutated_sequence not in mutations and mutated_sequence not in allWindows:
                    break
            mutations[mutated_sequence] = [numSegments] # unknown sequence is class last

        # update the allWindows dict with heavily mutated sequences
        allWindows.update(mutations)
        return numSegments+1, allWindows # adding in unknown as classifyable class


    def __mutate(self, sequence, probability):
        seq = list(sequence)
        rand = np.random.rand(len(sequence))
        mutIdx = np.argwhere(rand <= probability).flatten()
        for i in mutIdx:
            seq[i] = self.nucArr[random.randint(0, 3)]
        return "".join(seq)

    def __mutation_func(self, window, probability, how_many):
        mutations = []
        for i in range(how_many):
            mutations.append(self.__mutate(window, probability))
        return mutations

    def __merge_lists(self, list1, list2):
        # first list plus unique things in list2, not in list1
        resultant = list1 + list(set(list2) - set(list1))
        return resultant

    def __register_meta(self, key, value):
        if key not in self.meta:
            self.meta[key] = value

    @logger
    def _export_meta(self):
        np.save(self.directory +"/meta.npy", self.meta)
        self.serialization.save_meta(self.directory+"/serialization-meta.npy")

    @logger
    def _introduce_mutations(self, allWindows):
        allMutations = {}
        for window, segIds in allWindows.items():
            mutated_windows = self.__mutation_func(window, self.mutation_prob, self.mutation_freq)
            for mwindow in mutated_windows:
                if mwindow in allMutations:
                    allMutations[mwindow] = self.__merge_lists(segIds, allMutations[mwindow])
                elif mwindow in allWindows:
                    allMutations[mwindow] = self.__merge_lists(segIds, allWindows[mwindow])
                else:
                    allMutations[mwindow] = segIds

        # update the allWindows dict with allMutations
        allWindows.update(allMutations)
        return allWindows

    def _one_hot_input(self, window):
        bow = ku.encodeKmerBagOfWords(window, K=self.K)
        b = np.zeros(len(bow))
        b[np.argwhere(bow)] = 1
        return b

    def _one_hot_output(self, segIds, totalSegments):
        b = np.zeros(totalSegments)
        b[segIds] = 1
        return b

    @logger
    def _test_train_split(self, allWindows):
        train, test = train_test_split([*allWindows], test_size=0.30)
        return train, test

    @logger
    def _create_serializer(self):
        self.serialization = Serializer({'input': 'sparse', 'output': 'sparse'})

    @logger
    def write_tf(self, df, allWindows, numSegments, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for window in df:
                input = self._one_hot_input(window)
                output = self._one_hot_output(allWindows[window], numSegments)
                serializable_features = self.serialization.make_serializable(input=input, output=output)
                writer.write(serializable_features)

    @logger
    def create_fasta_mlp_minhash_interface(self):
        fasta = self.directory + "/sequence.fasta"
        fastamm = FastaMM(fasta, self.segment_length, self.window_length, self.K)
        fastamm.init()
        return fastamm

    def write(self):
        self.logger.info("Initiating...")
        fastamm = self.create_fasta_mlp_minhash_interface()
        numSegments, allWindows = self._read_windows_segments(fastamm)
        fastamm.writeMeta(self.directory)
        numSegments, allWindows = self._add_unknown_windows(numSegments, allWindows)
        allWindows = self._introduce_mutations(allWindows)
        train, test = self._test_train_split(allWindows)
        self.__register_meta('train_size', len(train))
        self.__register_meta('test_size', len(test))
        self.__register_meta('total', len(allWindows))
        self._create_serializer()
        self.write_tf(train, allWindows, numSegments, self.directory+"/train.tfrecords")
        del train
        self.write_tf(test, allWindows, numSegments, self.directory+"/test.tfrecords")
        del test, allWindows
        self._export_meta()


def reader_main():
    loader = Classification_Loader(FLAGS.data_dir, 1)
    features, labels = loader.load_dataset("test")
    with tf.Session() as sess:
        f,l = sess.run([features, labels])
        print("input", f, f.shape)
        print("output", l, l.shape)

def write_main():
    writer = Classification_Writer(FLAGS.data_dir)
    writer.write()

def main():
    write_main()
    # reader_main()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_classification_run/",
        help="Where is input data dir? use data_generation.py to create one")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()





