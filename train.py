import tensorflow as tf
from loadTFRecord import loadDataset, loadMeta
import numpy as np
import math

class Model:
    def __init__(self, input, label, batch_size):
        self.model_save_dir = 'model/'
        self.model_save_name = 'saved.ckpt'

        self.input_shape = input.shape
        self.label_shape = label.shape
        self.batch_size = batch_size
        self.model_params = []

        self.global_step = tf.Variable(10, trainable=False, name='global_step')
        self.logits = self._inference(input)
        self.loss = self._loss(self.logits, label)
        self.train_op = self._train(self.loss)
        self.evaluation, self.evaluate_op = self._evaluate(self.logits, label)

        self.model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.train.Saver(self.model_params, max_to_keep=4, keep_checkpoint_every_n_hours=2)

    def _inference(self, inputs):
        with tf.name_scope('fcc' ):
            d1 = self._make_dense(inputs, units=1500, activation=tf.nn.relu, name="layer_1")
            d1 = self._make_dense(d1, units=1000, activation=tf.nn.relu, name="layer_2")
            logits = self._make_dense(d1, units=self.label_shape[1], activation=None, name="layer_out")
            return logits

    def _make_dense(self, ip_tensor, units, activation, name):
        d1 = tf.layers.dense(inputs=ip_tensor, units=units, kernel_initializer=tf.truncated_normal_initializer, activation=activation, name=name)
        d1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        tf.summary.histogram('kernel', d1_vars[0])
        tf.summary.histogram('bias', d1_vars[1])
        tf.summary.histogram('act', d1)
        # self.model_params.append(d1.)
        # self.model_params.append(d1.bias)
        return d1

    def _loss(self, logits, labels):
        with tf.name_scope('loss'):
            # softmax = tf.nn.softmax(logits)
            # num_ones = tf.cast(tf.count_nonzero(labels, axis=1), dtype=tf.int32)
            # top_k_values, top_k_indices = tf.nn.top_k(softmax, num_ones)
            # print(top_k_values, top_k_indices)
            cast_labels = tf.cast(labels, dtype=tf.float32)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=cast_labels))
            # loss = tf.reduce_mean(top_k_values)
            tf.summary.scalar('loss', loss)
            # loss = tf.reduce_mean(tf.squared_difference(labels, logits))
            # loss = tf.cast(loss, dtype=tf.float32)
            # loss = tf.Print(loss, [loss], "loss =")
            # loss = tf.losses.softmax_cross_entropy(labels, logits=logits)
            return loss

    def _train(self, loss):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            return train_op

    def _evaluate(self, logits, labels):
        with tf.name_scope('evaluate'):
            precision, precision_op = tf.metrics.average_precision_at_k(labels, logits, k=2)
            # tf.summary.scalar('precision@2', precision)
            return precision, precision_op

            # softmax = tf.nn.softmax(logits)
            # # values_pred, indices_pred = tf.nn.top_k(softmax)
            # correct_prediction = tf.reduce_sum(softmax)
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # return accuracy
            # num_ones = tf.count_nonzero(labels)
            # top_k_values, top_k_indices = tf.nn.top_k(softmax, num_ones)


    def save(self, sess):
        self.saver.save(sess, self.model_save_dir + self.model_save_name, global_step=self.global_step)

    def restore(self, sess):
        path = tf.train.latest_checkpoint(self.model_save_dir)
        self.saver.restore(sess, path)


def main():
    batch_size = 100
    meta = loadMeta("run/meta.npy")
    print(meta)
    restore = False
    with tf.Graph().as_default():
        X, Y = loadDataset("run/train.tfrecords", batch_size)
        X_test, Y_test = loadDataset("run/test.tfrecords", batch_size, repeat=False)

        features = tf.placeholder(tf.float32, name="features", shape=[None, meta['feature_dense_shape'][0]])
        labels = tf.placeholder(tf.int64, name="labels", shape=[None, meta['label_dense_shape'][0]])
        model = Model(features,labels, batch_size)

        with tf.Session() as sess:
            if not restore:
                global_init = tf.global_variables_initializer()
                local_init = tf.local_variables_initializer()
                sess.run([global_init, local_init])
            else:
                model.restore(sess)

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("model/what/")
            writer.add_graph(sess.graph)

            for step in range(1000):
                _x,_y = sess.run([X,Y])
                summary, lossVal, _ = sess.run([merged, model.loss, model.train_op], feed_dict={features: _x, labels: _y})
                if step%100 == 0:
                    writer.add_summary(summary, step)
                    print("Batch Loss at step:", step, lossVal)
                if step%1000 == 0:
                    model.save(sess)

            while True:
                try:
                    _x_test, _y_test = sess.run([X_test, Y_test])
                    sess.run([model.evaluate_op, model.logits], feed_dict={features: _x_test, labels: _y_test})
                    # accuracy.append(testLoss)
                except tf.errors.OutOfRangeError:
                    break

            print(".....................Test Accuracy", sess.run([model.evaluation]))



if __name__ == "__main__":
    main()
