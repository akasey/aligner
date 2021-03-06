import tensorflow as tf

from loadTFRecord import Loader
from model import Model
import numpy as np
import sys
from common import *


def evaluate_f1_deprecated(labels, pred):
    intersection = np.sum(labels[pred==True])
    labels_count = np.sum(labels)
    pred_count = np.sum(pred)
    return 2*intersection/(labels_count + pred_count + 0.0)


def precision(labels, pred):
    intersection = np.sum(labels[pred==True])
    # labels_count = np.sum(labels)
    pred_count = np.sum(pred)
    return intersection / (pred_count + 0.0)

def recall(labels, pred):
    intersection = np.sum(labels[pred==True])
    labels_count = np.sum(labels)
    # pred_count = np.sum(pred)
    return intersection / (labels_count + 0.0)

def evaluate_f1(labels, pred):
    prec = precision(labels, pred)
    rec = recall(labels, pred)
    return 2*pred*rec/(pred + rec + 0.0)


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
        precision_total = 0
        recall_total = 0
        while True:
            try:
                _x_test, _y_test = sess.run([X_test, Y_test])
                activations = sess.run([layer_out], feed_dict={features: _x_test, labels: _y_test, dropout_keep_prob: 1.0})

                # precision monitor
                # k = 2
                for true_label, pred_label in zip(_y_test, activations[0]):
                    true_label, pred_label = np.array(true_label), sigmoid(np.array(pred_label))
                    precision_total += precision(true_label, pred_label)
                    recall_total += recall(true_label, pred_label)
                    print("True Label", true_label, "Pred Label", pred_label, "Precision", precision(true_label, pred_label), "Recall", recall(true_label, pred_label))
                    examplesCount += 1

            except tf.errors.OutOfRangeError:
                break
        print("Precision: ", precision_total/examplesCount, "Recall: ", recall_total/examplesCount)




if __name__ == "__main__":
    main(sys.argv)
