import tensorflow as tf
from loadTFRecord import Loader
import numpy as np
import math
import sys, os
from tensorflow.python.client import device_lib, timeline
from model import Model
from common import *

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def main(args):
    print("Devices list: ", device_lib.list_local_devices())

    if len(args) < 3:
        print("Usage: ", "<input_data_dir>", "<output_save_dir>")
        exit(0)
    dataDir = args[1]
    outputDir = args[2]
    batch_size = 512
    modelSaveDir = outputDir + '/model/'
    tensorboardDir = outputDir + '/tensorboard/'
    dumpDir = outputDir + '/dump/'
    if not os.path.exists(tensorboardDir):
        os.makedirs(tensorboardDir)
    if not os.path.exists(dumpDir):
        os.makedirs(dumpDir)
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)

    loader = Loader(dataDir, batch_size=batch_size)
    print(loader.meta)
    restore = True
    with tf.Graph().as_default():
        X, Y = loader.loadDataset("train")
        # X_test, Y_test = loader.loadDataset("test")

        features = tf.placeholder(tf.float32, name="features", shape=loader.getInputShape())
        labels = tf.placeholder(tf.int64, name="labels", shape=loader.getOutputShape())
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        model = Model(modelSaveDir, features,labels, dropout_keep_prob, batch_size, summaries=False)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            if not restore:
                print("Initializing network....")
                global_init = tf.global_variables_initializer()
                local_init = tf.local_variables_initializer()
                sess.run([global_init, local_init])
            else:
                print("Restoring network....")
                model.restore(sess)
                initialize_uninitialized(sess)
                local_init = tf.local_variables_initializer()
                sess.run([local_init])

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(tensorboardDir)
            writer.add_graph(sess.graph)

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            for step in range(150000):
                if restore:
                    _x,_y = sess.run([X,Y])
                    summary, lossVal, _ = sess.run([merged, model.loss, model.train_op],
                                                   feed_dict={features: _x, labels: _y, dropout_keep_prob: 0.7},
                                                   options=options, run_metadata=run_metadata)
                    writer.add_summary(summary, step)
                    print("Batch Loss at step:", step, lossVal)
                    if step % 1000 == 0:
                        model.save(sess)
                    # if (step+1) % 10 == 0:
                    #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    #     with open('timeline_nosummaries' + str(step) + '.json', 'w') as f:
                    #         f.write(chrome_trace)

"""
                accuracy_profile = []
                if (step+1) %200 == 0:
                    sess.run([local_init])
                    print("Evaluating accuracy.....")
                    while True:
                        try:
                            _x_test, _y_test = sess.run([X_test, Y_test])
                            _, act_out = sess.run([model.evaluate_op, model.logits],
                                                  feed_dict={features: _x_test, labels: _y_test, dropout_keep_prob: 0.7})

                            # precision monitor
                            k = 2
                            for true_label, pred_label in zip(_y_test, act_out):
                                true_label, pred_label = np.array(true_label), np.array(pred_label)
                                true_label_idx, pred_label_idx = true_label.argsort()[-k:][::-1],pred_label.argsort()[-k:][::-1]
                                accuracy_profile.append((true_label_idx, pred_label_idx))

                        except tf.errors.OutOfRangeError:
                            X_test, Y_test = loader.loadDataset("test")
                            break

                    with open(dumpDir + "/true","w") as trf, open(dumpDir+"/predict","w") as prf:
                        for tr,pr in accuracy_profile:
                            trf.write(str(tr) + "\n")
                            prf.write(str(pr) + "\n")

                    precision, eval_summary =  sess.run([model.evaluation, model.evaluation_summary])
                    print(".....................Test Precision",precision)
                    writer.add_summary(eval_summary, step)
"""


if __name__ == "__main__":
    main(sys.argv)
