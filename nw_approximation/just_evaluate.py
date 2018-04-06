import argparse
import tensorflow as tf
import numpy as np
from framework.config import Config
from data_reader import Loader
from multilayer_model import MultiLayerModel
from framework.common import make_logger

number_of_row = 8192
def getInputFn(data_dir):
    # loader = Loader(FLAGS.data_dir, batch_size=None)
    def input_fn():
        datapoints = tf.eye(65536)[2048:number_of_row+2048, :]
        points = tf.range(number_of_row)

        X = {'x1': datapoints, 'idx': points}
        return tf.data.Dataset.from_tensor_slices(X)

    return input_fn()

def read_kmers():
    dictionary = {}
    with open(FLAGS.data_dir+"/kmers.tsv") as fin:
        for idx,line in enumerate(fin.readlines()):
            dictionary[idx] = line.strip()
    return dictionary

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


def unwanted_stuffs(sess, model, loader, logger):
    logger.info("Loading training datasets..")
    features, labels = loader.load_dataset("train")
    logger.info("Making training op...")
    train_op, loss_op = model.train(features, labels)
    #
    # logger.info("Loading test datasets..")
    # features_eval, labels_eval = loader.load_dataset("test")
    #
    # eval_y = tf.placeholder(tf.float32, name="eval_labels", shape=model.get_output_shape())
    # logger.info("Making eval op...")
    # eval_mean_op, eval_update_op = model.evaluation(features_eval, labels_eval)
    #
    # summary_scalar_op = model.summary_scalars()
    # summary_histogram_op = model.summary_histograms()

def make_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

def main():
    logger = make_logger("just_evaluate")
    config = Config(FLAGS.model_dir + "/hparam.yaml")
    batch_size = config.training.get('batch_size', 128)
    loader = Loader(FLAGS.data_dir, config.training.get('batch_size', 128))
    logger.info("Making test dataset for reconstructing model...")
    features, labels = loader.load_dataset("test")
    logger.info("Making dataset iterator for prediction...")
    features_pred = getInputFn(FLAGS.data_dir)\
        .batch(512)\
        .make_one_shot_iterator().get_next()
    features_idx = features_pred['idx']

    with make_session() as sess:
        logger.info("Creating Multilayer model")
        model = MultiLayerModel(config)
        unwanted_stuffs(sess, model, loader, logger)
        do_init(sess, model)
        prediction_op = model.prediction(features_pred)

        logger.info("Creating Embedding matrix...")
        embedding = np.zeros([number_of_row, config.model['multilayer']['layer'][-1]])
        counter = 0
        logger.info("Into the loop...")
        while True:
            try:
                prediction, idx = sess.run([prediction_op, features_idx])
                embedding[idx] = prediction
                counter += batch_size
                logger.info("%d/%d" % (counter, number_of_row))
            except:
                break
                pass


    logger.info("Saving embedding matrix")
    # np.save("embedding.npy", embedding)
    logger.info("Computing similarity...")
    similarity = np.matmul(embedding, embedding.T)
    top_k = 20
    dictionary = read_kmers()
    for i in np.random.randint(0,number_of_row, 200):
        nearest = (-similarity[i, :]).argsort()[1:top_k + 1]
        print(dictionary[i], "close to", [dictionary[i] for i in nearest])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="fixed-2/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="scrap/",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()