import tensorflow as tf
from loadTFRecord import loadDataset, loadMeta

class Model:
    def __init__(self, input, label, batch_size):
        self.input_shape = input.shape
        self.label_shape = label.shape
        self.batch_size = batch_size

        self.logits = self._inference(input)
        self.loss = self._loss(self.logits, label)
        self.train_op = self._train(self.loss)
        self.evaluate_op = self._evaluate(self.logits, label)

    def _inference(self, inputs):
        with tf.name_scope('fcc' ):
            d1 = tf.layers.dense(inputs=inputs, units=200, activation=tf.nn.relu)
            d1 = tf.layers.dense(d1, units=150, activation=tf.nn.relu)
            d1 = tf.layers.dense(d1, units=100, activation=tf.nn.relu)
            logits = tf.layers.dense(d1, units=self.label_shape[1])
            return logits

    def _loss(self, logits, labels):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            # loss = tf.reduce_mean(tf.squared_difference(labels, logits))
            # loss = tf.cast(loss, dtype=tf.float32)
            # loss = tf.Print(loss, [loss], "loss =")
            # loss = tf.losses.softmax_cross_entropy(labels, logits=logits)
            return loss

    def _train(self, loss):
        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(2e-2)
            train_op = optimizer.minimize(loss)
            return train_op

    def _evaluate(self, logits, labels):
        with tf.name_scope('evaluate'):


    # def save(self, dir):



def main():
    batch_size = 100
    meta = loadMeta("run/meta.npy")
    print(meta)
    with tf.Graph().as_default():

        X, Y = loadDataset("run/train.tfrecords", batch_size)
        X_test, Y_test = loadDataset("run/test.tfrecords", batch_size, repeat=False)

        features = tf.placeholder(tf.float32, name="features", shape=[None, meta['feature_dense_shape'][0]])
        labels = tf.placeholder(tf.float32, name="labels", shape=[None, meta['label_dense_shape'][0]])
        model = Model(features,labels, batch_size)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for step in range(10000):
                _x,_y = sess.run([X,Y])
                lossVal, _ = sess.run([model.loss, model.train_op], feed_dict={features: _x, labels: _y})
                if step%100 == 0:
                    print("Batch Loss at step:", step, lossVal)
                # if i%100 == 0:
                #     testLoss = sess.run()
                #     print(".................Test Loss", testLoss)

        writer = tf.summary.FileWriter("model/what/")
        writer.add_graph(sess.graph)

if __name__ == "__main__":
    main()
