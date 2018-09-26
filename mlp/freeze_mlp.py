import argparse
import tensorflow as tf
import os
import math
from framework import freeze_graph
from framework.config import Config
from classification_loader import Classification_Loader
from tensorflow.python.tools import optimize_for_inference_lib



def main():
    model_save_location = FLAGS.model_dir + "/frozen-graphs/"
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)

    model_checkpoint_location = FLAGS.model_dir + "/tensorboard"
    input_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_location)
    input_graph_path = model_checkpoint_location +"/model.pb"
    output_graph_path = model_save_location + "/frozen_graph.pb"
    input_saver_def_path = ""
    input_binary = False

    # Note that we this normally should be only "output_node"!!!
    input_node_names = "inputs"
    output_node_names = "output_logits"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = True

    print("Freezing the graph....")
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")

    ### further optimization
    print("Further optimizing the graph....")
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_graph_path, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_names], [output_node_names],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    ### writing meta for C++ to read
    print("Writing meta for aligner..")
    with tf.Session() as sess:
        config = Config(FLAGS.model_dir + "/hparam.yaml")
        loader = Classification_Loader(FLAGS.data_dir, config.training.get('batch_size', 1))
        features, labels = loader.load_dataset("train")

        dictionary = {
            "input_layer" : "inputs",
            "output_layer" : "output_logits",
            "input_shape" : features.shape[-1].value,
            "output_shape" : labels.shape[-1].value,
            "K" : int(math.log(features.shape[-1].value, 4))
        }
        with open(model_save_location + "/aligner.meta", "w") as fout:
            for k,v in dictionary.items():
                fout.write( "%s : %s\n" % (str(k), str(v)) )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="sample_classification_run/longWindow/model_dir/",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_classification_run/longWindow",
        help="Where is input data dir? use data_generation.py to create one")


    FLAGS, unparsed = parser.parse_known_args()
    main()