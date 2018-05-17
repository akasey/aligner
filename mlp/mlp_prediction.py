import argparse
import numpy as np
import tensorflow as tf
import os

from framework.common import make_logger, make_session
from framework.config import Config
from classification_loader import Classification_Loader
from autoencoder.encoder_writer import Kmer_Utility as ku
from mlp import MultiLayerPerceptron

def do_init(sess, model):
    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    if model.restore(sess):
        initialize_uninitialized(sess)
        local_init = tf.local_variables_initializer()
        sess.run([local_init])

class Read:
    def __init__(self, sequence, position):
        self.sequence = sequence
        self.position = position
        self.pred_segment = None
        self.pred_position = None

complementMap = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
def rev_complement(sequence):
    toRet = ""
    for s in reversed(sequence):
        toRet += complementMap[s]
    return toRet

def read_fastq(filename):
    reads = []
    with open(filename, "r") as fin:
        start, sequence,forward = -1, "", True
        for idx, line in enumerate(fin.readlines()):
            if idx % 4==0:
                splits = line.split("_")
                if "rand" in splits[0]:
                    start = -123
                else:
                    start = splits[1]
                forward = (splits[3] == "0")
            elif idx%4 == 1:
                sequence = line.strip() if forward else rev_complement(line.strip())
            elif idx %4 == 3:
                # print(start, sequence)
                reads.append(Read(sequence, start))
                start, sequence, forward = -1, "", True
    return reads

def make_unit_one_hot(read):
    bow = ku.encodeKmerBagOfWords(read.sequence, K=FLAGS.k)
    b = np.zeros(len(bow))
    b[np.argwhere(bow)] = 1
    return np.reshape(b, (-1,len(bow)))




def main():
    logger = make_logger("MLP.Prediction")
    logger.info("Reading fastq")
    reads = read_fastq(FLAGS.fastq)
    logger.info("Complete reading fastq")
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    device = config.training.get('device', '/cpu:0')
    model = MultiLayerPerceptron(config)

    with make_session(device) as sess:
        # unwanted_stuffs(sess, model, loader, logger)
        X = tf.placeholder(tf.float32, shape=(None, 4**FLAGS.k))
        pred_op = tf.round(model.prediction(X))
        logger.info("Restoring model")
        do_init(sess, model)
        logger.info("Restoring restored")

        fout = open(FLAGS.out_dir + "/predictions", "w")
        total = len(reads)
        counter = 0

        for read in reads:
            one_hot = make_unit_one_hot(read)
            prediction = sess.run(pred_op, feed_dict={X: one_hot})
            prediction = np.reshape(prediction, (-1,))
            prediction = np.argwhere(prediction).flatten()
            fout.write(read.sequence + "$$" + str(read.position) + "$$" + ",".join([str(x) for x in prediction]) + "\n")
            counter += 1
            if counter % 1000 == 0:
                logger.info(str(counter) + "/" + str(total))

        fout.close()





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--fastq",
        type=str,
        default="sample_classification_run/out.bwa.read1.fastq",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_classification_run/model_dir/1000/",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sample_classification_run/",
        help="Path for storing the model checkpoints.")
    FLAGS, unparsed = parser.parse_known_args()

    main()