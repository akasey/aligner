import argparse
import tensorflow as tf
import numpy as np
import random
import concurrent.futures as futures
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
from framework.config import Config
from mlp import MultiLayerPerceptron
from classification_loader import Classification_Loader
from autoencoder.encoder_writer import Kmer_Utility as ku


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def ReLU(x):
    return x * (x > 0)

def numpyArrayToCSVLine(x):
    return ",".join([str(i) for i in x])

def idx_to_kmer(idx, K):
    nucMap = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    kmerString = ''
    while idx > 0:
        rem = idx % 4
        kmerString = kmerString + nucMap[rem]
        idx = idx // 4

    while len(kmerString) != K:
        kmerString = kmerString + nucMap[0]

    return kmerString

def _mutations(idx_list, num, K=7):
    ret = []
    nucMap = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    for idx in idx_list:
        kmer = idx_to_kmer(idx, K)
        for i in range(num):
            ret.append(ku.encodeKmer(idx_to_kmer(idx+4*i+1, K)))
    return ret



def make_columnlist(K):
    columns = []
    for i in range(4**K):
        columns.append(idx_to_kmer(i, K))
    columns.append('strand')
    return columns

def main():
    config = Config(FLAGS.model_dir+"/hparam.yaml")
    model = MultiLayerPerceptron(config)
    loader = Classification_Loader(FLAGS.data_dir, config.training.get('batch_size', 512))
    K = np.log(config.input_features -1)/np.log(4)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        features, labels = loader.load_dataset("test")
        train_op, loss_op = model.train(features, labels)
        if model.restore(sess):
            initialize_uninitialized(sess)
            local_init = tf.local_variables_initializer()
            sess.run([local_init])
            weight_var1 = [v for v in tf.trainable_variables() if v.name == "mlp/layer_0_weights:0"][0]
            bias_var1 = [v for v in tf.trainable_variables() if v.name == "mlp/layer_0_biases:0"][0]
            weight_var2 = [v for v in tf.trainable_variables() if v.name == "mlp/layer_1_weights:0"][0]
            bias_var2 = [v for v in tf.trainable_variables() if v.name == "mlp/layer_1_biases:0"][0]
            weights1, bias1, weights2, bias2 = sess.run([weight_var1, bias_var1, weight_var2, bias_var2])

            columnList = make_columnlist(int(K))
            inputVector = pd.DataFrame(columns=columnList)
            firstLayer = pd.DataFrame(columns=list(range(weight_var1.shape[1])))
            secondLayer = pd.DataFrame(columns=list(range(weight_var2.shape[1])))

            counter = 0
            while counter < loader.test_size:
                input = sess.run([features])
                input = np.squeeze(np.array(input))
                activation1 = np.matmul(input,weights1) + bias1
                activation1 = ReLU(activation1)
                activation2 = np.matmul(activation1, weights2) + bias2
                activation2 = ReLU(activation2)
                for i in range(input.shape[0]):
                    inputVector.loc[counter] = input[i]
                    firstLayer.loc[counter] = activation1[i]
                    secondLayer.loc[counter] = activation2[i]
                    counter += 1

                if counter > 2000:
                    break
            # inputVector.to_hdf(FLAGS.data_dir+"/inputVector.h5", 'inputVector', format='table', mode='w')
            inputVector.to_pickle(FLAGS.data_dir+"/inputVector.pkl", compression="gzip")
            # firstLayer.to_hdf(FLAGS.data_dir+"/firstLayer.h5", 'firstLayer', format='table', mode='w')
            firstLayer.to_pickle(FLAGS.data_dir+"/firstLayer.pkl", compression="gzip")
            # secondLayer.to_hdf(FLAGS.data_dir+"/secondLayer.h5", 'secondLayer', format='table', mode='w')
            secondLayer.to_pickle(FLAGS.data_dir+"/secondLayer.pkl", compression="gzip")


            activations = []
            for a in ['AAAAAAA','CAAAAAA','CCCCCCC']:
                v = ku.encodeKmer(a)
                vector = np.zeros((weights1.shape[0]))
                vector[v] = 1
                print(vector.shape, weights1.shape)
                activation = np.matmul(vector,weights1) + bias1
                activation = ReLU(activation)
                activations.append(activation)
                print(activation, activation.shape)
            print(activations)


def correlation():
    input = pd.read_pickle(FLAGS.data_dir + "/inputVector.pkl", compression="gzip")
    first_layer = pd.read_pickle(FLAGS.data_dir + "/firstLayer.pkl", compression="gzip")

    def task(series1, series2, series1name, series2name):
        return series1name, series2name, abs(series1.corr(series2))
        # return series1name, series2name, jaccard_similarity_score(series1, series2)

    df = pd.DataFrame(columns=["col1", "col2", "correlation"])
    pool = futures.ThreadPoolExecutor(FLAGS.threads)

    randint = [random.randint(0,16385) for i in range(5)]
    additional = _mutations(randint, 10)
    randint = randint + additional
    randint = sorted(randint)

    xyz = 0
    for col2 in first_layer:
        xyz += 1
        tasks_future = []
        counter = 0
        if xyz > 500:
            break
        for col1 in input:
            counter += 1
            if counter in randint:
                s1name, s2name, corr = task(input[col1], first_layer[col2], col1, col2)
                df = df.append({'col1': s1name, 'col2': s2name, 'correlation':corr}, ignore_index=True)
        x = 1

    df.to_pickle(FLAGS.data_dir + "/correlation.pkl", compression="gzip")


def analyse():
    correlation = pd.read_pickle(FLAGS.data_dir + "/activation_count_second_layer.pkl", compression="gzip")
    # correlation = pd.read_pickle(FLAGS.data_dir + "/activation_count.pkl", compression="gzip")
    # kmers = ['TAAAATA', 'AAAATAT', 'AAATATT', 'AATATTT', 'ATATTTT', 'TATTTTT', 'ATTTTTT', 'TTTTTTT', 'TTTTTTA'] + \
    #         ['AAAAAAA', 'AAAAAAA', 'AAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT', 'TTTTTTT'] + \
    #         ['TAAAATG', 'AAAATGT', 'AAATGTT', 'AATGTTT', 'ATGTTTT', 'TGTTTTT', 'GTTTTTT', 'TTTTTTT', 'TTTTTTA'] + \
    #         ['ACCAAAA', 'CCAAAAA', 'CAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT', 'TTTTTTT']
    kmers = ['TAGAAAT', 'AGAAATT', 'GAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTC', 'TTTTTCT', 'TTTTCTT', 'TTTCTTG'] + \
            ['AAAAAAA', 'AAAAAAA', 'AAAAAAA', 'AAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT'] + \
            ['TAGAAAT', 'AGAAATT', 'GAAATTT', 'AAATTTA', 'AATTTAA', 'ATTTAAT', 'TTTAATT', 'TTAATTC', 'TAATTCT'] + \
            ['AACATAA', 'ACATAAA', 'CATAAAA', 'ATAAAAT', 'TAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT']
            # ['TCGAAAT', 'CGAAATT', 'GAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTC', 'TTTTTCT', 'TTTTCTT', 'TTTCTTG'] + \
            # ['AAAGGAA', 'AAGGAAA', 'AGGAAAA', 'GGAAAAT', 'GAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT']

    # kmers = ['TACAAAT', 'ACAAATT', 'CAAATTT', 'AAATTTT', 'AATTTTC', 'ATTTTCC', 'TTTTCCT', 'TTTCCTT', 'TTCCTTG'] + \
    #         ['AACATAA', 'ACATAAA', 'CATAAAA', 'ATAAAAT', 'TAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT']
    kmers = []
    for k in ['GTTCTGATGACAATG', 'TTCTGATGACAATGA', 'GTAGAGGTAAAAGTG', 'TGGCTGTTGTTGGTG', 'CTGTTGTTGGTGTTC']:
        kmers = kmers + all_kmers(k, 7)
    neurons = set(correlation["neuron"])


    X = np.zeros((len(kmers), len(neurons)))
    Y = np.zeros((len(kmers), len(neurons)))
    Z = np.zeros((len(kmers), len(neurons)))
    for idx, k in enumerate(kmers):
        for i, neuron in enumerate(neurons):
            df = correlation[(correlation.kmer==k) & (correlation.neuron==neuron)]
            X[idx, i] = df["sum"] if len(df)!=0 else 0
            Y[idx, i] = df["count"] if len(df)!=0 else 0
            Z[idx, i] = df["average_act"] if len(df)!=0 else 0


    correlation


def simply_count():
    def insert_or_update(df, kmer, neuron, sum, count):
        if ((df['kmer'] == kmer) & (df['neuron'] == neuron)).any():
            row = df[(df['kmer'] == kmer) & (df['neuron'] == neuron)]
            df.loc[(df['kmer'] == kmer) & (df['neuron'] == neuron)] = [kmer, neuron, sum+row['sum'], count+row['count']]
        else:
            df = df.append({'kmer': kmer, 'neuron': neuron, 'sum': sum, "count": count}, ignore_index=True)
        return df


    input = pd.read_pickle(FLAGS.data_dir + "/inputVector.pkl", compression="gzip")
    first_layer = pd.read_pickle(FLAGS.data_dir + "/secondLayer.pkl", compression="gzip")
    first_layer = first_layer[first_layer.columns[0:1000]]
    counter_sum = {}
    counter_count = {}

    for input_row, act_row in zip(input.iterrows(), first_layer.iterrows()):
        input_row_idx = input_row[0]
        input_row = input_row[1]
        act_row_idx = act_row[0]
        act_row = act_row[1]

        non_zero_input = input_row[input_row>0]
        non_zero_act = act_row[act_row>0]
        for kmer in non_zero_input.keys():
            for idx, act in non_zero_act.iteritems():
                key = kmer + '-' + str(idx)
                counter_sum[key] = act + (0 if key not in counter_sum else counter_sum[key])
                counter_count[key] = 1 + (0 if key not in counter_count else counter_count[key])

    i=0;
    df = pd.DataFrame(columns=["kmer", "neuron", "sum", "count", "average_act"])
    for sum_key, sum_value in counter_sum.items():
        counter_value = counter_count[sum_key]
        kmer = sum_key.split("-")[0]
        if counter_value > 10:
            neuron = int(sum_key.split("-")[1])
            # df = df.append({'kmer': kmer, 'neuron': neuron, 'sum': sum_value, "count": counter_value}, ignore_index=True)
            df.loc[i] = {'kmer': kmer, 'neuron': neuron, 'sum': sum_value, "count": counter_value, "average_act": sum_value/counter_value}
            i+=1

    df.to_pickle(FLAGS.data_dir + "/activation_count_second_layer.pkl", compression="gzip")


def all_kmers(str, K):
    all = []
    for i in range(len(str)-K+1):
        all.append(str[i:i+K])
    return all #list(set(all))

def unique_kmer():
    K = 15
    topN = 5

    def readGenome():
        file = open(FLAGS.data_dir +"/sequence.fasta", "r")
        genome = ""
        for line in file.readlines():
            if '>' not in line:
                genome += line.strip()
        return genome

    def topThings(X, idx1, idx2):
        difference = X[:, idx1] - X[:, idx2]
        difference_sorted_idx = difference.argsort()
        topThingsInIdx1 = difference_sorted_idx[-topN:][::-1]
        topThingsInIdx2 = difference_sorted_idx[:topN]
        topKmersInIdx1 = [(idx_to_kmer(i, K),str(X[i, idx1])+'..'+str(X[i,idx2])) for i in topThingsInIdx1]
        topKmersInIdx2 = [(idx_to_kmer(i, K),str(X[i, idx1])+'..'+str(X[i,idx2])) for i in topThingsInIdx2]
        print("In", idx1, topKmersInIdx1)
        print("In", idx2, topKmersInIdx2)
        print()

    genome = readGenome()
    segments = [(0,50099), (49900, 100099), (99900, 150099), (149900, 200099)]
    X = np.zeros((4**K, len(segments)), dtype=np.int8)
    for idx, segment in enumerate(segments):
        genemic_section = genome[segment[0]: segment[1]]
        for i in range(len(genemic_section)-K+1):
            kmer = ku.encodeKmer(genemic_section[i:i+K])
            X[kmer, idx] += 1

    topThings(X, 0, 1)
    topThings(X, 1, 2)
    topThings(X, 2, 3)
    topThings(X, 1, 3)

    """
In 0 [('TTTTTTTTTTTTTTT', '5..0'), ('TAAAAAAAATATATT', '2..0'), ('AAAAAGTTTTAAAAA', '2..0'), ('TATTTTAAAAATTTT', '2..0'), ('TTTTAAAAAAATTTT', '2..0')]
In 1 [('CAAATTAATAAAAAA', '0..2'), ('TTAAATAAAAAAAAA', '0..2'), ('TTTAATAAAAATAAT', '0..2'), ('AAAATATTTTTTTAA', '0..2'), ('AAAATTTTAAAAATT', '0..2')]

In 1 [('TTTTTTAAAAAAAAA', '2..0'), ('ATTAAAATATTTTTT', '2..0'), ('TAAAAAAAAAATTAT', '2..0'), ('AAAATATTTTTTTAA', '2..0'), ('ATAATATAAATTTAG', '2..0')]
In 2 [('ACAGTTGCAGCCACC', '0..1'), ('TACTCTACTTAATTA', '0..1'), ('AGGCAGTCTCTTATT', '0..1'), ('CTTAACGCGTTAGCT', '0..1'), ('TTTAATTCTTCTTTC', '0..1')]

In 2 [('TAGAAATTTTTCTTG', '1..0'), ('TTTCAAGGTCGACTA', '1..0'), ('TTTAGACTGGTATTT', '1..0'), ('CTTTTCGTTACTAAA', '1..0'), ('TGCTGCCTTCCTTAG', '1..0')]
In 3 [('AAAAAAAAATTTTTT', '0..3'), ('AAAAAAAATTTTTTT', '0..3'), ('ATATTTTTTTAAAAA', '0..3'), ('AAAAATTTTTTTTTT', '0..3'), ('AAAAAAATTTTTTTT', '0..3')]

In 1 [('TAAAATATTTTTTTA', '2..0'), ('AATGAAAAAAATTTT', '2..0'), ('ATGTTTTAAAAATAT', '2..0'), ('TTAGAAATAATAAAT', '2..0'), ('ATATTAAAATTAAAA', '2..0')]
In 3 [('AAAAAAAATTTTTTT', '0..3'), ('AAAAAAATTTTTTTT', '0..3'), ('AAAAAAAAATTTTTT', '0..3'), ('AAAAATTTTTTTTTT', '0..3'), ('TAAAAAAATTTTTTT', '0..2')]
    """

    x=1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/yeast/model_dir/8000_4000",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/akash/PycharmProjects/aligner/sample_classification_run/yeast",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Threads")

    FLAGS, unparsed = parser.parse_known_args()
    try:
        sess = tf.Session()
    except:
        pass
    # main()
    # correlation()
    # simply_count()
    # unique_kmer()
    analyse()
    # good examples taken from index[0] between 2-3.. Line 271,272
    print(all_kmers("TAGAAATTTTTCTTG", 7)) # ['TAGAAAT', 'AGAAATT', 'GAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTC', 'TTTTTCT', 'TTTTCTT', 'TTTCTTG']
    print(all_kmers("AAAAAAAAATTTTTT", 7)) # ['AAAAAAA', 'AAAAAAA', 'AAAAAAA', 'AAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT']
    # print(all_kmers("TCGAAATTTTTCTTG", 7)) # ['TCGAAAT', 'CGAAATT', 'GAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTC', 'TTTTTCT', 'TTTTCTT', 'TTTCTTG']
    # print(all_kmers("AAAGGAAAATTTTTT", 7)) # ['AAAGGAA', 'AAGGAAA', 'AGGAAAA', 'GGAAAAT', 'GAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT']

    # print(all_kmers("TAGAAATTTAATTCT", 7))
    # print(all_kmers("AACATAAAATTTTTT", 7))

    print(all_kmers("TACAAATTTTCCTTG", 7))
    print(all_kmers("AACATAAAATTTTTT", 7))
    print()

    # not so good difference
    print(all_kmers("TAAAATATTTTTTTA", 7)) # ['TAAAATA', 'AAAATAT', 'AAATATT', 'AATATTT', 'ATATTTT', 'TATTTTT', 'ATTTTTT', 'TTTTTTT', 'TTTTTTA']
    print(all_kmers("AAAAAAAATTTTTTT", 7)) # ['AAAAAAA', 'AAAAAAA', 'AAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT', 'TTTTTTT']
    print(all_kmers("TAAAATGTTTTTTTA", 7)) # ['TAAAATG', 'AAAATGT', 'AAATGTT', 'AATGTTT', 'ATGTTTT', 'TGTTTTT', 'GTTTTTT', 'TTTTTTT', 'TTTTTTA']
    print(all_kmers("ACCAAAAATTTTTTT", 7)) # ['ACCAAAA', 'CCAAAAA', 'CAAAAAT', 'AAAAATT', 'AAAATTT', 'AAATTTT', 'AATTTTT', 'ATTTTTT', 'TTTTTTT']



# Yeast
# In 0 [('GTTCTGATGACAATG', '10..0'), ('TTCTGATGACAATGA', '10..0'), ('GTAGAGGTAAAAGTG', '10..0'), ('TGGCTGTTGTTGGTG', '9..0'), ('CTGTTGTTGGTGTTC', '9..0')]
# In 1 [('GAAGAAGAAGAAGAA', '0..6'), ('AAGAAGAAGAAGAAG', '0..5'), ('AGAAGAAGAAGAAGA', '0..5'), ('AAGGAAAAGGAGAAG', '0..3'), ('TTTTCTTTTTTTTTT', '0..3')]
#
# In 1 [('GAAGAAGAAGAAGAA', '6..0'), ('AAGAAGAAGAAGAAG', '5..0'), ('AGAAGAAGAAGAAGA', '5..0'), ('AAGGAAAAGGAGAAG', '3..0'), ('AAGGAAAAAAAGGAA', '2..0')]
# In 2 [('GTTGTTGTTGTTGTT', '0..5'), ('TTGTTGTTGTTGTTG', '0..4'), ('TGTTGTTGTTGTTGT', '0..4'), ('AGAGAGAGAGAGAGA', '0..4'), ('GAGAGAGAGAGAGAG', '0..3')]
#
# In 2 [('AAAAAAAAAAAAAAA', '10..0'), ('GTTGTTGTTGTTGTT', '5..0'), ('AGAGAGAGAGAGAGA', '4..0'), ('TTGTTGTTGTTGTTG', '4..0'), ('TGTTGTTGTTGTTGT', '4..0')]
# In 3 [('TTTTTTTTTTTTTTT', '1..12'), ('TACTACTACTACTAC', '0..6'), ('CTACTACTACTACTA', '0..5'), ('ACTACTACTACTACT', '0..5'), ('AAATGTGGCTCTTCC', '0..3')]
#
# In 1 [('AAAAAAAAAAAAAAA', '8..0'), ('GAAGAAGAAGAAGAA', '6..0'), ('AAGAAGAAGAAGAAG', '5..0'), ('AGAAGAAGAAGAAGA', '5..0'), ('TTTTCTTTTTTTTTT', '3..0')]
# In 3 [('TTTTTTTTTTTTTTT', '1..12'), ('TACTACTACTACTAC', '0..6'), ('CTACTACTACTACTA', '0..5'), ('ACTACTACTACTACT', '0..5'), ('TCGACGAACCCAGTG', '0..3')]