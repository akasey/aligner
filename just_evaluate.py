import tensorflow as tf

from loadTFRecord import Loader
from model import Model
import numpy as np
import sys
from common import *


def evaluate_f1(labels, pred):
    intersection = np.sum(labels[pred==True])
    labels_count = np.sum(labels)
    pred_count = np.sum(pred)
    return 2*intersection/(labels_count + pred_count + 0.0)


def main(args):
    if len(args) != 3:
        print("Usage: ", "<data_dir> <model_saved_dir>")
        exit(0)

    dataDir = args[1]
    modelDir = args[2]
    batch_size = 512
    loader = Loader(dataDir, batch_size=batch_size)
    X_test, Y_test = loader.loadDataset("test")

    # chkp.print_tensors_in_checkpoint_file(path, tensor_name='', all_tensors=True, all_tensor_names=True)
    # print(path)
    with tf.Session() as sess:
        path = tf.train.latest_checkpoint(modelDir+"/model")
        saver = tf.train.import_meta_graph(path+'.meta')
        saver.restore(sess, path)
        # print(tf.get_default_graph().as_graph_def())

        graph = tf.get_default_graph()
        features = graph.get_tensor_by_name("features:0")
        labels = graph.get_tensor_by_name("labels:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

        layer_out = graph.get_tensor_by_name("fcc/layer_out_activation:0")
        print(layer_out)

        examplesCount = 0
        precision = 0
        recall = 0
        while True:
            try:
                _x_test, _y_test = sess.run([X_test, Y_test])
                activations = sess.run([layer_out], feed_dict={features: _x_test, labels: _y_test, dropout_keep_prob: 1.0})

                # precision monitor
                # k = 2
                for true_label, pred_label in zip(_y_test, activations[0]):
                    true_label, pred_label = np.array(true_label), np.array(pred_label)
                    # true_label_idx, pred_label_idx = true_label.argsort()[-k:][::-1], pred_label.argsort()[-k:][::-1]
                    # print("True Label", true_label, "Pred Logits", pred_label)
                    print("True Label", true_label, "Pred Label", sigmoid(pred_label))
                    examplesCount += 1

            except tf.errors.OutOfRangeError:
                break




if __name__ == "__main__":
    main(sys.argv)
