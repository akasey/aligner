import tensorflow as tf
from loadTFRecord import Loader
import numpy as np
import math
import sys, os

class Model:
    def __init__(self, output_dir, input, label, batch_size):
        self.model_save_dir = output_dir
        self.model_save_name = 'model.ckpt'

        self.input_shape = input.shape
        self.label_shape = label.shape
        self.batch_size = batch_size
        self.model_params = []

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.logits = self._inference(input)
        self.loss = self._loss(self.logits, label)
        self.train_op = self._train(self.loss)
        self.evaluation, self.evaluation_summary, self.evaluate_op = self._evaluate(self.logits, label)

        self.saver = tf.train.Saver(self.model_params, max_to_keep=2)

    def _inference(self, inputs):
        with tf.name_scope('fcc' ):
            d1 = self._make_dense(inputs, units=1500, activation_fn=tf.nn.relu, name="layer_1")
            d1 = self._make_dense(d1, units=1000, activation_fn=tf.nn.relu, name="layer_2")
            d1 = self._make_dense(d1, units=500, activation_fn=tf.nn.relu, name="layer_3")
            d1 = self._make_dense(d1, units=200, activation_fn=tf.nn.relu, name="layer_4")
            logits = self._make_dense(d1, units=self.label_shape[1].value, activation_fn=None, name="layer_out")
            return logits

    def _make_dense(self, ip_tensor, units, activation_fn, name):
        shape = [ip_tensor.shape[1].value, units]
        weights = self.get_weights(shape, name)
        biases = self.get_biases(shape, name)
        activation = tf.matmul(ip_tensor, weights) + biases
        if activation_fn:
            activation = activation_fn(activation)
        tf.summary.histogram(name + "_weights", weights)
        tf.summary.histogram(name + "_biases", biases)
        tf.summary.histogram(name + "_act", activation)
        self.model_params.append(weights)
        self.model_params.append(biases)
        return activation

    def _loss(self, logits, labels):
        with tf.name_scope('loss'):
            cast_labels = tf.cast(labels, dtype=tf.float32)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=cast_labels))
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=cast_labels))
            tf.summary.scalar('loss', loss)
            return loss

    def _train(self, loss):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            return train_op

    def _evaluate(self, logits, labels):
        with tf.name_scope('evaluate'):
            precision, precision_op = tf.metrics.average_precision_at_k(labels, logits, k=2)
            summary_op = tf.summary.scalar('precision@2', precision)
            return precision, summary_op, precision_op


    def save(self, sess):
        self.saver.save(sess, self.model_save_dir + self.model_save_name, global_step=self.global_step)

    def restore(self, sess):
        path = tf.train.latest_checkpoint(self.model_save_dir)
        self.saver.restore(sess, path)

    def get_weights(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform(shape, minval=(-std), maxval=std), name=(name + "_weights"))

    def get_biases(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform([shape[-1]], minval=(-std), maxval=std), name=(name + "_biases"))


def main(args):
    if len(args) < 3:
        print("Usage: ", "<input_data_dir>", "<output_save_dir>")
        exit(0)
    dataDir = args[1]
    outputDir = args[2]
    batch_size = 512
    modelSaveDir = outputDir + '/model/'
    tensorboardDir = outputDir + '/tensorboard/'
    dumpDir = outputDir + '/dump/'
    if not os.path.exists(tensorboardDir):
        os.makedirs(tensorboardDir)
    if not os.path.exists(dumpDir):
        os.makedirs(dumpDir)
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)

    loader = Loader(dataDir, batch_size=batch_size)
    print(loader.meta)
    restore = False
    with tf.Graph().as_default():
        X, Y = loader.loadDataset("train")
        X_test, Y_test = loader.loadDataset("test")

        features = tf.placeholder(tf.float32, name="features", shape=loader.getInputShape())
        labels = tf.placeholder(tf.int64, name="labels", shape=loader.getOutputShape())
        model = Model(modelSaveDir, features,labels, batch_size)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if not restore:
                print("Initializing network....")
                global_init = tf.global_variables_initializer()
                local_init = tf.local_variables_initializer()
                sess.run([global_init, local_init])
            else:
                print("Restoring network....")
                model.restore(sess)
                local_init = tf.local_variables_initializer()
                sess.run([local_init])

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tensorboardDir)
            writer.add_graph(sess.graph)

            for step in range(10000):
                if not restore:
                    _x,_y = sess.run([X,Y])
                    summary, lossVal, _ = sess.run([merged, model.loss, model.train_op], feed_dict={features: _x, labels: _y})
                    writer.add_summary(summary, step)
                    print("Batch Loss at step:", step, lossVal)
                    if step % 100 == 0:
                        model.save(sess)

                accuracy_profile = []
                if (step+1) %200 == 0:
                    sess.run([local_init])
                    print("Evaluating accuracy.....")
                    while True:
                        try:
                            _x_test, _y_test = sess.run([X_test, Y_test])
                            _, act_out = sess.run([model.evaluate_op, model.logits],
                                                  feed_dict={features: _x_test, labels: _y_test})

                            # precision monitor
                            k = 2
                            for true_label, pred_label in zip(_y_test, act_out):
                                true_label, pred_label = np.array(true_label), np.array(pred_label)
                                true_label_idx, pred_label_idx = true_label.argsort()[-k:][::-1],pred_label.argsort()[-k:][::-1]
                                accuracy_profile.append((true_label_idx, pred_label_idx))

                        except tf.errors.OutOfRangeError:
                            X_test, Y_test = loader.loadDataset("test")
                            break

                    with open(dumpDir + "/true","w") as trf, open(dumpDir+"/predict","w") as prf:
                        for tr,pr in accuracy_profile:
                            trf.write(str(tr) + "\n")
                            prf.write(str(pr) + "\n")

                    precision, eval_summary =  sess.run([model.evaluation, model.evaluation_summary])
                    print(".....................Test Precision",precision)
                    writer.add_summary(eval_summary, step)



if __name__ == "__main__":
    main(sys.argv)
