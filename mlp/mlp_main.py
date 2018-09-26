import argparse
import os

from framework.trainer import TrainExecuter
from framework.common import make_logger
from framework.config import Config
from classification_loader import Classification_Loader
from mlp import MultiLayerPerceptron

import tensorflow as tf

def sanity_check():
    assert os.path.exists(FLAGS.model_dir+"/hparam.yaml"), FLAGS.model_dir+"/hparam.yaml" + " not found"

def main():
    logger = make_logger("Main.AutoEncoder")
    sanity_check()
    config = Config(FLAGS.model_dir+"/hparam.yaml")
    loader = Classification_Loader(FLAGS.data_dir, config.training.get('batch_size', 512))
    logger.info("Creating Autoencoder model")
    model = MultiLayerPerceptron(config)
    logger.info("Creating TrainExecutor")
    executer = TrainExecuter(config)
    logger.info("TrainExecutor.run().....")
    executer.run(model, loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_classification_run/longWindow",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_classification_run/longWindow/model_dir/",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    main()