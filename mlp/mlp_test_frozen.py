import argparse
import tensorflow as tf
import numpy as np
from autoencoder.encoder_writer import Kmer_Utility as ku

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph

def read_meta(metafile):
    toRet = dict()
    with open(metafile, "r") as fin:
        for line in fin.readlines():
            split = line.split(":")
            toRet[split[0].strip()] = split[1].strip()
    return toRet


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/Users/akash/ClionProjects/aligner_minhash/cmake-build-debug/ecoli/model_dir/frozen-graphs/", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.model_dir + "frozen_graph.pb")
    meta = read_meta(args.model_dir + "aligner.meta")
    print(meta)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
        # print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = graph.get_tensor_by_name("import/" + meta["input_layer"] + ":0")
    y = graph.get_tensor_by_name("import/" + meta["output_layer"] + ":0")

    # We launch a Session
    k = int(meta["K"])
    with tf.Session(graph=graph) as sess:
        # while True:
        #     sequence = input("Enter Sequence: ")
            sequence = "TTCATGGAGTATTTCTGCTTTAGTCGGCAATGTTGCACGCAAGGATCCGAGAGCGTCACTACGCGCCGGTTATCAGCATAAAACCGAGTAAGGTAGACGCTACCTGACCTGTACGTCACAGCGAGACCGTAATGAAGCCCGGAAAGCCGGTTTGAATGGTTCATGCACTGCGACGGGGGAGGATACCGCTTTCCTCGAGT"
            bow = ku.encodeKmerBagOfWords(sequence, K=k)
            b = np.zeros(len(bow) + 1)
            b[np.argwhere(bow)] = 1
            b[-1] = 0
            y_out = sess.run(y, feed_dict={
                x: [b]
            })
            result = np.argwhere(np.round(y_out))
            print ("Result", result)
