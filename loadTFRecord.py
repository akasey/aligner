import tensorflow as tf
import numpy as np


class Loader:
    def __init__(self, dirname, batch_size):
        self.dirname = dirname
        self.batch_size = batch_size
        self.meta = self.__loadMeta(self.dirname+"/meta.npy")
        self.trainDataset = None
        self.testDataset = None

    def __loadMeta(self, filename):
        metaTemp = np.load(filename)
        meta = {}
        for k in metaTemp.item():
            meta[k] = metaTemp.item().get(k)
        return meta

    def __serializedToRows(self, serializedThing):
        example_features = {'feature': tf.SparseFeature(index_key=['feature_indices_0', 'feature_indices_1'],
                                                          value_key='feature_values',
                                                          dtype=tf.int64,
                                                          size=self.meta['feature_dense_shape']),
                            'label': tf.SparseFeature(index_key=['label_indices_0', 'label_indices_1'],
                                                        value_key='label_values',
                                                        dtype=tf.int64,
                                                        size=self.meta['label_dense_shape']) }

        rows = tf.parse_single_example(
            serializedThing,
            features=example_features)
        feature, label = rows['feature'], rows['label']
        feature, label = tf.reshape(tf.sparse_tensor_to_dense(feature), [self.meta['feature_dense_shape'][0]]), tf.reshape(tf.sparse_tensor_to_dense(label), [self.meta['label_dense_shape'][0]])
        feature, label = tf.cast(feature, dtype=tf.float32), tf.cast(label, dtype=tf.int64)

        return feature, label


    def __loadDataset(self, filename, batch_size, repeat=True):
        with tf.name_scope('loadDataset'):
            dataset = tf.data.TFRecordDataset(self.dirname + "/" + filename).prefetch(4*batch_size)
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.map(map_func=self.__serializedToRows, num_parallel_calls=20) \
                        .batch(batch_size)
                        # .shuffle(1024)\

            return dataset

    def loadDataset(self, type):
        assert type in ["test", "train"], "Type should be 'train' or 'test'"
        if type == "train":
            if not self.trainDataset:
                self.trainDataset = self.__loadDataset("train.tfrecords", batch_size=self.batch_size)
            return self.trainDataset.make_one_shot_iterator().get_next()
        elif type == "test":
            if not self.testDataset:
                self.testDataset = self.__loadDataset("test.tfrecords", batch_size=self.batch_size, repeat=False)
            return self.testDataset.make_one_shot_iterator().get_next()

    def getInputShape(self):
        return [None, self.meta['feature_dense_shape'][0]]

    def getOutputShape(self):
        return [None, self.meta['label_dense_shape'][0]]


if __name__ == "__main__":
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    loader = Loader("run/", batch_size=100)
    X, Y = loader.loadDataset("train")
    # X, Y = tf.sparse_tensor_to_dense(X), tf.sparse_tensor_to_dense(Y)

    with sess.as_default():
        for i in range(1):
            _x, _y = sess.run([X,Y])
            print(_x.shape)
            print(_y.shape)
