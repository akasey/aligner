import argparse
import tensorflow as tf
import numpy as np

from nw_approx.model import Encoder_Model
from nw_approx.pretrain_reader import PreTrainLoader
from framework.common import make_logger
from framework.config import Config


def make_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

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

def main():
    logger = make_logger("nw_approx_evaluate")
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    batch_size = 1#config.training.get('batch_size', 128)
    projection_layer_size = config.model['multilayer']['layer'][-1]
    input_shape = config.input_features
    model = Encoder_Model(config)
    loader = PreTrainLoader(FLAGS.data_dir, batch_size)
    features, labels = loader.load_dataset("test")
    device = config.training.get('device', '/cpu:0')

    total = 5000
    actual_data = np.zeros([total, input_shape], dtype=np.float32)
    embedding = np.zeros([total,projection_layer_size], dtype=np.float32)
    with make_session(device) as sess:
        unwanted_stuffs(sess, model, loader, logger)
        prediction_op = model.prediction(features)
        prediction_op = tf.nn.sigmoid(prediction_op)
        do_init(sess, model)

        for i in range(total):
            inputs, predictions = sess.run([features, prediction_op])
            actual_data[i,:] = inputs
            embedding[i,:] = predictions

        embedding_tensor = tf.placeholder(tf.float32, shape=[total,projection_layer_size])
        embedding_matrix = tf.get_variable("embedding_matrix", [total,projection_layer_size], initializer=tf.constant_initializer(value=0.0))
        actual_data_tensor = tf.placeholder(tf.float32, shape=[total,input_shape])
        actual_matrix = tf.get_variable("actual_matrix", [total, input_shape], initializer=tf.constant_initializer(value=0.0))

        init_op = embedding_matrix.assign(embedding_tensor), actual_matrix.assign(actual_data_tensor)

        _ = sess.run(init_op, feed_dict={actual_data_tensor: actual_data,
                                                                    embedding_tensor: embedding})

        saver = tf.train.Saver([actual_matrix, embedding_matrix])
        saver.save(sess, FLAGS.model_dir+'/evaluation_things/model.ckpt', 0)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_autoencoder_run/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_autoencoder_run/5000_3000_1500/",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()