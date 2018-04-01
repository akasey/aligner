import os
import argparse
import tensorflow as tf
from config import Config
from multilayer_model import MultiLayerModel
from data_reader import Loader
from trainer import TrainExecuter
from common import make_logger

def sanity_check():
    assert os.path.exists(FLAGS.model_dir+"/hparam.yaml"), FLAGS.model_dir+"/hparam.yaml" + " not found"

def get_input_fn():
    pass

def main():
    logger = make_logger("Main")
    sanity_check()
    config = Config(FLAGS.model_dir+"/hparam.yaml")
    loader = Loader(FLAGS.data_dir, config.training.get('batch_size', 512))
    logger.info("Creating Multilayer model")
    model = MultiLayerModel(config)
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