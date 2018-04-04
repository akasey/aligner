import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

workDir = "sample_autoencoder_run"
WINDOW_LEN = 1000
nucMap = {'A':0, 'C':1, 'G':2, 'T':3}
k=6

kmerEncoding = {}

class Kmer_Utility:
    @staticmethod
    def encodeKmer(kmer):
        if kmer in kmerEncoding:
            return kmerEncoding[kmer]
        numeric = 0
        for idx, nt in enumerate(kmer):
            numeric += nucMap[nt] * 4**idx
        kmerEncoding[kmer] = numeric
        return numeric

    @staticmethod
    def readGenome(fasta):
        lines = ""
        with open(fasta, "r") as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    lines += line.strip()
        return lines

    @staticmethod
    def slidingWindow(segment, winlength):
        windows = []
        for i in range(len(segment)-winlength+1):
            window = segment[i:i+winlength]
            windows.append(window)
        return windows

    @staticmethod
    def encodeKmerBagOfWords(window, last_window=None, last_encoded_window=None):
        # incremental -- Not sure if dictionary order is preserved
        if last_window is not None and last_encoded_window is not None and last_window[1:] == window[0:-1]:
            last_start_kmer = last_window[0:0 + k]
            # last_encoded_window[encodeKmer(last_start_kmer)] = max(0, last_encoded_window[encodeKmer(last_start_kmer)]-1)
            last_encoded_window[Kmer_Utility.encodeKmer(last_start_kmer)] -= 1
            this_end_kmer = window[-k:]
            last_encoded_window[Kmer_Utility.encodeKmer(this_end_kmer)] += 1
            return last_encoded_window

        # sliding kmer extraction
        b = np.zeros(4 ** k)
        for i in range(len(window) - k + 1):
            kmer_num = Kmer_Utility.encodeKmer(window[i:i + k])
            b[kmer_num] += 1
        return b.flatten()

    @staticmethod
    def sparseRepresentation(arr):
        indices0 = np.where(arr)[0]
        indices1 = np.zeros(np.shape(indices0)[0], dtype=np.int8)
        values = np.ones(np.shape(indices0)[0], dtype=np.int8)
        dense_shape = np.array([np.shape(arr)[0], 1])
        return indices0, indices1, values, dense_shape


def makeRow(encoding):
    indices_0, indices_1, values, dense_shape = Kmer_Utility.sparseRepresentation(encoding)
    row = {
        'indices_0': indices_0,
        'indices_1': indices_1,
        'values': values,
        'dense_shape': dense_shape
    }
    return row

def writeTFRecord(df, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for row in df:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=row['indices_0'])),
                        'indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=row['indices_1'])),
                        'values': tf.train.Feature(int64_list=tf.train.Int64List(value=row['values'])),
                    }))
            writer.write(example.SerializeToString())

if __name__=="__main__":
    print("Reading genome..")
    genome = Kmer_Utility.readGenome(workDir+"/sequence.fasta")
    print("Computing all windows..")
    windows = Kmer_Utility.slidingWindow(genome, winlength=WINDOW_LEN)
    print("Total %d windows..." % len(windows))
    last_encoded_window = None
    last_window = None
    df = []
    meta = {}
    for window in windows:
        encoding = Kmer_Utility.encodeKmerBagOfWords(window, last_window=last_window, last_encoded_window=last_encoded_window)
        last_window = window
        last_encoded_window = encoding
        row = makeRow(encoding)
        df.append(row)
        if len(meta) == 0:
            meta['input_shape'] = row['dense_shape']

    meta['total'] = len(df)
    print("Test train splits..")
    train, test = train_test_split(df, test_size=0.20)
    meta['train_size'] = len(train)
    meta['test_size'] = len(test)
    print("Writing train tfrecords..")
    writeTFRecord(train, workDir+"/train.tfrecords")
    print("Writing test tfrecords..")
    writeTFRecord(test, workDir + "/test.tfrecords")
    np.save(workDir + "/meta.npy", meta)


