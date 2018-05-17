import tensorflow as tf
import numpy as np

class PreTrainLoader():
    def __init__(self, dirname, batch_size):
        self.dirname = dirname
        self.batch_size = batch_size
        self.trainDataset = None
        self.testDataset = None
        self.meta = self._get_meta(dirname+"/pretrain-meta.npy")

    def _serializedToRows(self, serializedThing):
        example_features = {'x1': tf.SparseFeature(index_key=['kmer1_indices_0', 'kmer1_indices_1'],
                                                        value_key='kmer1_values',
                                                        dtype=tf.int64,
                                                        size=[self.meta['input_shape'][0], 1]),
                            'x2': tf.SparseFeature(index_key=['kmer2_indices_0', 'kmer2_indices_1'],
                                                        value_key='kmer2_values',
                                                        dtype=tf.int64,
                                                        size=[self.meta['input_shape'][0], 1]),
                            'score': tf.FixedLenFeature([1], tf.float32)}

        rows = tf.parse_single_example(
            serializedThing,
            features=example_features)
        x1, x2, score = rows['x1'], rows['x2'], rows['score']
        x1, x2 = tf.sparse_tensor_to_dense(x1), tf.sparse_tensor_to_dense(x2)
        x1,x2, score = tf.reshape(x1, [self.meta['input_shape'][0], ]), tf.reshape(x2, [self.meta['input_shape'][0], ]), tf.reshape(score, [1,])
        x1, x2, score = tf.cast(x1, dtype=tf.float32), tf.cast(x2, dtype=tf.float32), tf.cast(score, dtype=tf.float32)

        feature = {'x1':x1, 'x2':x2}

        return feature,score

    def _load_dataset(self, filename, batch_size, repeat=True):
        with tf.name_scope('loadDataset'):
            dataset = tf.data.TFRecordDataset(filename).prefetch(4 * batch_size)
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.map(map_func=self._serializedToRows, num_parallel_calls=20) \
                .batch(batch_size) \
                .shuffle(buffer_size=500)
            return dataset

    def _get_meta(self, filename):
        metaTemp = np.load(filename)
        meta = {}
        for k in metaTemp.item():
            meta[k] = metaTemp.item().get(k)
        return meta

    def load_dataset(self):
        if not self.trainDataset:
            self.trainDataset = self._load_dataset(self.dirname + "/pretrain.tfrecords", batch_size=self.batch_size)
        return self.trainDataset.make_one_shot_iterator().get_next()

    def getInputShape(self):
        return [None, self.meta['input_shape'][0]]