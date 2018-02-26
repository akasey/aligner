import tensorflow as tf
import numpy as np

def loadMeta(filename):
    metaTemp = np.load(filename)
    meta = {}
    for k in metaTemp.item():
        meta[k] = metaTemp.item().get(k)
    return meta

meta = loadMeta('run/meta.npy')

def serializedToRows(serializedThing):
    example_features = {'feature': tf.SparseFeature(index_key=['feature_indices_0', 'feature_indices_1'],
                                                      value_key='feature_values',
                                                      dtype=tf.int64,
                                                      size=meta['feature_dense_shape']),
                        'label': tf.SparseFeature(index_key=['label_indices_0', 'label_indices_1'],
                                                    value_key='label_values',
                                                    dtype=tf.int64,
                                                    size=meta['label_dense_shape']) }
    # example_features= {'feature_indices': tf.VarLenFeature(dtype=tf.int64),
    #                    'label_indices': tf.VarLenFeature(dtype=tf.int64) }
    rows = tf.parse_single_example(
        serializedThing,
        features=example_features)
    feature, label = rows['feature'], rows['label']
    # x = tf.Print(feature, [feature, label], "x")
    feature, label = tf.sparse_tensor_to_dense(feature), tf.sparse_tensor_to_dense(label)
    # feature = tf.Print(feature, [feature, tf.shape(feature), label, tf.shape(label)],' printing values ')
    return feature, label

def reshaping(feature, label):
    print("reshaping_1", feature, label)
    # f,l = tf.sparse_tensor_to_dense(feature), tf.sparse_tensor_to_dense(label)
    # f = tf.reshape(f, [-1,meta['feature_dense_shape'][0]])
    # l = tf.reshape(l, [-1,meta['label_dense_shape'][0]])

    f,l = feature, label
    f,l = tf.reshape(f, [meta['feature_dense_shape'][0]]), tf.reshape(l, [meta['label_dense_shape'][0]])

    f, l = tf.cast(f, dtype=tf.float32), tf.cast(l, dtype=tf.int64)
    print("reshaping_2", f, l)
    return f, l


def loadDataset(filename, batch_size, repeat=True):
    with tf.name_scope('loadDataset'):
        dataset = tf.data.TFRecordDataset(filename)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(serializedToRows)\
            .map(reshaping)\
            .shuffle(10000)\
            .batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


if __name__ == "__main__":
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    X, Y = loadDataset("run/train.tfrecords", 2)
    # X, Y = tf.sparse_tensor_to_dense(X), tf.sparse_tensor_to_dense(Y)

    with sess.as_default():
        for i in range(1):
            _x, _y = sess.run([X,Y])
            print(_x.shape)
            print(_y.shape)