import tensorflow as tf
import logging
import math
from common import make_logger

class TrainExecuter():
    def __init__(self, config):
        self.config = config
        self.batch_size = config.training.get('batch_size', 512)
        self.epoch = config.training.get('epoch', 15)
        self.eval_frequency = config.training.get('eval_frequency', 100)
        self.log_frequency = config.training.get('log_frequency', 10)
        self.histogram_frequency = config.training.get('histogram_frequency', 1000)
        self.device = config.training.get('device', '/cpu:0')
        self.logger = make_logger("Train Executor")
        self.restore = True

    def run(self, model, loader):
        writer = self.make_writer()

        with self.make_session() as sess:
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.logger.info("Loading training datasets..")
            features, labels = loader.load_dataset("train")
            self.logger.info("Making training op...")
            train_op, loss_op = model.train(features, labels)

            self.logger.info("Loading test datasets..")
            features_eval, labels_eval = loader.load_dataset("test")

            self.eval_y = tf.placeholder(tf.float32, name="eval_labels", shape=model.get_output_shape())
            self.logger.info("Making eval op...")
            eval_mean_op, eval_update_op = model.evaluation(features_eval, labels_eval)

            summary_scalar_op = model.summary_scalars()
            summary_histogram_op = model.summary_histograms()

            max_steps = math.ceil(loader.getTotal() / self.batch_size) * self.epoch
            self.do_init(sess, model)
            writer.add_graph(sess.graph)
            self.logger.info("Initializaton complete now running into loop")
            for step in range(max_steps):
                sess.run([train_op])

                if (step+1) % self.histogram_frequency == 0:
                    self.logger.info("Saving histograms..")
                    hist_stat = sess.run(summary_histogram_op)
                    writer.add_summary(hist_stat, global_step=step)
                    self.logger.info("Saving model at step: %d" % (step))
                    model.save(sess)

                # time for evaluation
                if (step+1) % self.eval_frequency == 0:
                    self.evaluate(sess, eval_update_op, loader)
                    stat = sess.run(summary_scalar_op)
                    writer.add_summary(stat, global_step=step)
                    eval_mean_score = sess.run(eval_mean_op)
                    self.logger.info("Evaluation at step %d : %f" % (step, eval_mean_score))
                if step % self.log_frequency == 0:
                    stat, loss = sess.run([summary_scalar_op, loss_op])
                    writer.add_summary(stat, global_step=step)
                    self.logger.info("Loss at step %d/%d: %f" % (step, max_steps, loss))

            self.logger.info("Finishing up...")
            hist_stat, scalar_stat, loss = sess.run([summary_histogram_op, summary_scalar_op, loss_op])
            writer.add_summary(hist_stat, global_step=step)
            writer.add_summary(scalar_stat, global_step=step)
            writer.add_summary(loss, global_step=step)
            model.save(sess)
            self.logger.info("Finished...")


    def make_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    def make_writer(self):
        writer = tf.summary.FileWriter(self.config.runtime['model_dir'] + "/tensorboard", flush_secs=120)
        return writer

    def evaluate(self, sess, eval_update_op, loader):
        counter = 0
        while counter < loader.test_size:
            act_out = sess.run([eval_update_op])
            counter += self.batch_size
            if (counter *100 / loader.test_size) % 10 == 0:
                self.logger.info("Evaluating %d/%d" % (counter, loader.test_size))

    def do_init(self, sess, model):
        def initialize_uninitialized(sess):
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        if self.restore:
            self.logger.info("Restoring network....")
            if model.restore(sess):
                initialize_uninitialized(sess)
                local_init = tf.local_variables_initializer()
                sess.run([local_init])
            else:
                self.logger.info("Restoring failed...")
                self.restore = False

        if not self.restore:
            self.logger.info("Initializing network....")
            global_init = tf.global_variables_initializer()
            local_init = tf.local_variables_initializer()
            sess.run([global_init, local_init])