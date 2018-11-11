import tensorflow as tf
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from framework.config import Config
from classification_loader import Classification_Loader
from mlp import MultiLayerPerceptron


def do_init(sess, model):
    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    restore = True
    if restore:
        if model.restore(sess):
            initialize_uninitialized(sess)
            local_init = tf.local_variables_initializer()
            sess.run([local_init])
        else:
            restore = False

    if not restore:
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        sess.run([global_init, local_init])

def make_predictions():
    print("Making inference...")
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    model = MultiLayerPerceptron(config)
    loader = Classification_Loader(FLAGS.data_dir, config.training.get('batch_size', 512))
    Y_true = []
    Y_pred = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=2)) as sess:
        features, labels = loader.load_dataset("test")
        prediction_op = model.prediction(features)
        predicted_labels = tf.round(prediction_op)
        do_init(sess, model)

        counter = 0
        while counter < loader.test_size:
            predictions, true_labels = sess.run([predicted_labels, labels])
            for pred, true in zip(predictions, true_labels):
                Y_true.append(true)
                Y_pred.append(pred)
            counter += loader.batch_size
            if (counter//100) % 100 == 0:
                print(counter, "/", loader.test_size)
                # break

    return Y_true, Y_pred

def micro_average(Y_true, Y_pred):
    Y, Z, YZ = 0.0, 0.0, 0.0
    for i in range(Y_true.shape[1]):
        YZ += np.sum(Y_true[:, i] * Y_pred[:, i])
        Y += np.sum(Y_true[:, i])
        Z += np.sum(Y_pred[:, i])

    print("Micro-averaged", "Precision: ", YZ/Z, "Recall: ", YZ/Y)


def main():
    Y_true, Y_pred = make_predictions()
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    print(len(Y_true))
    micro_average(Y_true, Y_pred)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(Y_true.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(Y_true[:, i],
                                                            Y_pred[:, i])
        average_precision[i] = average_precision_score(Y_true[:, i], Y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_true.ravel(),
                                                                    Y_pred.ravel())
    average_precision["micro"] = average_precision_score(Y_true, Y_pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # plt.figure()
    # plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #         .format(average_precision["micro"]))
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/model_dir",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Threads")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass

    main()