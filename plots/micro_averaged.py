import tensorflow as tf
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from framework.config import Config
from framework.common import make_logger
from lstm.dataset import LSTMClassificationLoader
from mlp.classification_loader import Classification_Loader
from mlp.mlp import MultiLayerPerceptron
from lstm.lstm import LSTM_Classification


logger = make_logger("MicroAveraging")

def do_init(sess, model):
    logger.info("Restoring model")
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
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    if 'lstm' not in config.model:
        model = MultiLayerPerceptron(config)
        loader = Classification_Loader(FLAGS.data_dir, config.training.get('batch_size', 512))
    elif 'lstm' in config.model:
        model = LSTM_Classification(config)
        loader = LSTMClassificationLoader(FLAGS.data_dir, config.training.get('batch_size', 512))
    Y_true = []
    Y_pred = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=2)) as sess:
        features, labels = loader.load_dataset("test")
        prediction_op = model.prediction(features)
        predicted_labels = tf.round(prediction_op)
        do_init(sess, model)

        logger.info("Making inference...")
        counter = 0
        while counter < loader.test_size:
            predictions, true_labels = sess.run([predicted_labels, labels])
            for pred, true in zip(predictions, true_labels):
                Y_true.append(true)
                Y_pred.append(pred)
            counter += loader.batch_size
            if (counter//100) % 100 == 0:
                logger.info("Progress: %d/%d" % (counter, loader.test_size))
                # break

    return Y_true, Y_pred

def micro_average(Y_true, Y_pred):
    Y, Z, YZ = 0.0, 0.0, 0.0
    for i in range(Y_true.shape[1]):
        YZ += np.sum(Y_true[:, i] * Y_pred[:, i])
        Y += np.sum(Y_true[:, i])
        Z += np.sum(Y_pred[:, i])

    logger.info("Micro-averaged:: Precision: %f, Recall: %f" %(YZ/Z, YZ/Y))


def main():
    Y_true, Y_pred = make_predictions()
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    logger.info("Test size: %d, NumClasses: %d" % (Y_true.shape[0], Y_true.shape[1]))
    micro_average(Y_true, Y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/model_dir",
        # default="/Users/akash/PycharmProjects/aligner/sample_lstm_run/model_dir",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/",
        # default="/Users/akash/PycharmProjects/aligner/sample_lstm_run/",
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