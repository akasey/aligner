import argparse
import tensorflow as tf
import numpy as np

from autoencoder import AutoEncoder
from framework.common import make_logger, make_session
from framework.config import Config
from denoising_loader import DenoisingLoader


def unwanted_stuffs(sess, model, loader, logger):
    logger.info("Loading training datasets..")
    features, labels = loader.load_dataset("train")
    logger.info("Making training op...")
    train_op, loss_op = model.train(features, labels)

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

def activation(x):
    sigmoid = tf.sigmoid(x)
    pullup = sigmoid + tf.ones_like(sigmoid)*0.2
    return tf.round(pullup)


def main():
    logger = make_logger("Evaluation.Denoising")
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    device = config.training.get('device', '/cpu:0')
    batch_size = 1#config.training.get('batch_size', 128)
    model = AutoEncoder(config)
    loader = DenoisingLoader(FLAGS.data_dir, batch_size)
    features, labels = loader.load_dataset("test")

    with make_session(device) as sess:
        unwanted_stuffs(sess, model, loader, logger)
        _,_ = model.evaluation(features, labels)
        do_init(sess, model)
        output = activation(model.eval_output_logit)

        counter = 0
        sum = 0
        while True:
            input, logits, lab = sess.run([features, output, labels])
            print(np.sum(logits-lab))
            sum += np.sum(logits-lab)
            counter += 1
            if counter == loader.test_size:
                break

        print("Total sum", sum)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_denoising_run/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_denoising_run/model_dir/",
        help="Where is input data dir? use data_generation.py to create one")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()