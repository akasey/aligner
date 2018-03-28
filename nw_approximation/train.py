import argparse
import os
import numpy as np
import math
import tensorflow as tf
import logging
import pickle
from data_reader import Loader


logging.getLogger().setLevel(logging.INFO)

def model_fn(features, labels, mode, params):

    model_params = []

    def _get_weights(shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.get_variable(name+"_weights", shape, initializer=tf.random_uniform_initializer(-std, std))

    def _get_biases(shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.get_variable(name + "_biases", [shape[-1]], initializer=tf.random_uniform_initializer(-std, std))

    def _make_dense(ip_tensor, units, activation_fn, name):
        shape = [ip_tensor.shape[1].value, units]
        with tf.device(params.devicePrior):
            weights = _get_weights(shape, name)
            biases = _get_biases(shape, name)
            if activation_fn:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases)
                activation = activation_fn(activation, name=name + "_activation")
                activation = tf.nn.dropout(activation, keep_prob=params.dropOutKeep)
            else:
                activation = tf.add(tf.matmul(ip_tensor, weights), biases, name=name + "_activation")

        if params.summaries:
            tf.summary.histogram(name + "_weights", weights)
            tf.summary.histogram(name + "_biases", biases)
            tf.summary.histogram(name + "_act", activation)

        model_params.append(weights)
        model_params.append(biases)
        return activation

    def _encoder(inputs):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            layer = inputs
            for idx, layer_size in enumerate(params.layers):
                layer = _make_dense(layer, layer_size, name="layer_"+(str(idx) if idx != len(params.layers)-1 else "out"), activation_fn=tf.nn.relu if idx != len(params.layers)-1 else None)
            return layer

    x1 = _encoder(features['x1'])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": x1, "features": features['x1']}
        )

    else:
        x2 = _encoder(features['x2'])
        l2distance = tf.reduce_sum(tf.square(tf.subtract(x1, x2)), axis=1)
        # sqrt_l2 = tf.sqrt(l2distance)
        # squared_difference = tf.subtract(sqrt_l2, labels)
        squared_difference = tf.square(tf.subtract(l2distance, tf.square(labels)))
        loss = tf.reduce_mean(squared_difference)
        evaluation = tf.metrics.mean(squared_difference)


        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=params.gradient_clipping_norm,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": x1},
            loss=loss,
            train_op=train_op,
            eval_metric_ops={"mean_difference": evaluation})

def input_fn(loader, mode):

    def input_fn():
        if mode == tf.estimator.ModeKeys.TRAIN:
            return loader.load_dataset("train")
        else:
            return loader.load_dataset("test")
    return input_fn

def create_estimator_and_specs(run_config, model_params):
    loader = Loader(model_params.data_dir, model_params.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn(
        loader=loader,
        mode=tf.estimator.ModeKeys.TRAIN), max_steps=model_params.max_steps
    )

    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn(
        loader=loader,
        mode=tf.estimator.ModeKeys.EVAL))

    return estimator, train_spec, eval_spec

def restore_model_params(flags):
    if os.path.exists(flags.model_dir + "/hparams"):
        import pickle
        logging.getLogger().info("Restoring hyperparameters from %s" % (flags.model_dir + "/hparams"))
        fin = open(flags.model_dir + "/hparams", 'rb')
        protobuf = pickle.load(fin)
        model_params = tf.contrib.training.HParams(hparam_def=protobuf)
        return model_params
    return None

def save_model_params(hparams):
    import pickle
    protobuf = hparams.to_proto()
    with open(hparams.model_dir+"/hparams", "wb") as fout:
        pickle.dump(protobuf, fout)

def get_model_params(restore=True):
    if restore:
        model_params = restore_model_params(flags=FLAGS)
        if not model_params:
            restore=False

    if not restore:
        logging.getLogger().info("Creating new hyperparameters for later")
        loader = Loader(FLAGS.data_dir, None)
        model_params = tf.contrib.training.HParams(
            devicePrior='/cpu:0',
            dropOutKeep=0.7,
            summaries=False,
            layers=[100,50],
            learning_rate=FLAGS.learning_rate,
            gradient_clipping_norm=9.0,
            batch_size=FLAGS.batch_size,
            max_steps=math.ceil(loader.getTotal() / FLAGS.batch_size)*FLAGS.num_epoch,
            data_dir=FLAGS.data_dir,
            model_dir=FLAGS.model_dir)

    for k,v in model_params.values().items():
            print(k,": ",v)
    return model_params

def main():
    hparams = get_model_params()
    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=hparams.model_dir,
            save_checkpoints_steps=500,
            keep_checkpoint_max=2,
            save_summary_steps=0),
        model_params=hparams)
    save_model_params(hparams)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="fixed-2/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=1,
        help="Total epochs to train")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runtime_junk/",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training.")

    FLAGS, unparsed = parser.parse_known_args()
    main()