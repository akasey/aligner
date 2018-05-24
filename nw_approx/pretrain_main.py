import tensorflow as tf
import argparse

from framework.common import make_logger, make_session
from framework.config import Config
from nw_approx.pretrain_reader import PreTrainLoader
from nw_approx.model import Encoder_Model


logger = make_logger("Pretrain")


def make_writer():
    writer = tf.summary.FileWriter(FLAGS.model_dir + "/tensorboard", flush_secs=120)
    return writer

def do_init(sess, model):
    restore = True
    def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    if restore:
        logger.info("Restoring network....")
        if model.restore(sess):
            initialize_uninitialized(sess)
            local_init = tf.local_variables_initializer()
            sess.run([local_init])
        else:
            logger.info("Restoring failed...")
            restore = False

    if not restore:
        logger.info("Initializing network....")
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        sess.run([global_init, local_init])

def run(model, loader, config):
    writer = make_writer()
    histogram_frequency = config.training.get('histogram_frequency', 100)
    log_frequency = config.training.get('log_frequency', 10)

    with make_session(config.training.get('device', '/cpu:0')) as sess:
        global_step = tf.Variable(0, trainable=False, name='global_step')
        logger.info("Loading training datasets..")
        features, labels = loader.load_dataset()
        logger.info("Making training op...")
        train_op, loss_op = model.train(features, labels)
        for i in range(2):
            x,y = sess.run([features,labels])


        summary_scalar_op = model.summary_scalars()
        summary_histogram_op = model.summary_histograms()

        max_steps = 10000*config.training.get('epoch', 20)
        do_init(sess, model)
        writer.add_graph(sess.graph)
        logger.info("Initializaton complete now running into loop")
        for step in range(max_steps):
            sess.run([train_op])

            if (step + 1) % histogram_frequency == 0:
                logger.info("Saving histograms..")
                hist_stat = sess.run(summary_histogram_op)
                writer.add_summary(hist_stat, global_step=step)
                logger.info("Saving model at step: %d" % (step))
                model.save(sess)

            # time for evaluation
            if step % log_frequency == 0:
                stat, loss = sess.run([summary_scalar_op, loss_op])
                writer.add_summary(stat, global_step=step)
                logger.info("Loss at step %d/%d: %f" % (step, max_steps, loss))

        logger.info("Finishing up...")
        hist_stat, scalar_stat, loss = sess.run([summary_histogram_op, summary_scalar_op, loss_op])
        writer.add_summary(hist_stat, global_step=step)
        writer.add_summary(scalar_stat, global_step=step)
        logger.info("Final loss: %f" % loss)
        model.save(sess)
        logger.info("Finished...")


def main():
    config = Config(FLAGS.model_dir+"/hparam.yaml")
    model = Encoder_Model(config)
    loader = PreTrainLoader(FLAGS.data_dir, config.training.get('batch_size', 128))
    run(model, loader, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_nw_approx_run/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_nw_approx_run/model_dir/",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()