import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from autoencoder.encoder_writer import Kmer_Utility
import time

outDir = "sample_nw_approx_run/"
WINDOW_LEN = 1000
nucMap = {'A':0, 'C':1, 'G':2, 'T':3}
k=6
FIXED_POINTS = 100
MATCH = 1
MISMATCH = -1
GAP = -1

kmerEncoding = {}


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

def needleman_wunsch(str1, str2):
    m = len(str1) +1
    n = len(str2) +1
    str1 = "-" + str1
    str2 = "+" + str2
    matrix = np.ones((m,n)) * -12312413241
    for i in range(m):
        for j in range(n):
            direction = [(-1, -1,MATCH if str1[i]==str2[j] else MISMATCH), (-1, 0, GAP), (0, -1, GAP)]
            coord = [(i+x, j+y, z) for (x,y,z) in direction]
            values = [matrix[x,y]+z if (x>-1 and y>-1) else -12312413241 for x,y,z in coord]
            matrix[i,j] = 0 if i==0 and j==0 else max(values)
    return matrix[-1,-1]

def write_fixed_points(fixed_windows):
    meta = {}
    print("Doing fixed points...")
    counter = 0
    with tf.python_io.TFRecordWriter(outDir+"pretrain.tfrecords") as writer, open(outDir+"distance.csv", "w", 10) as fout:
        for idx1, win1 in enumerate(fixed_windows):
            bow_win1 = Kmer_Utility.encodeKmerBagOfWords(win1)
            bow1_indices_0, bow1_indices_1, bow1_values, bow1_dense_shape = Kmer_Utility.sparseRepresentation(bow_win1)
            for idx2, win2 in enumerate(fixed_windows):
                bow_win2 = Kmer_Utility.encodeKmerBagOfWords(win2)
                bow2_indices_0, bow2_indices_1, bow2_values, bow2_dense_shape = Kmer_Utility.sparseRepresentation(bow_win2)
                score = euclidean_distance(bow_win1, bow_win2)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'kmer1_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1_indices_0)),
                            'kmer1_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1_indices_1)),
                            'kmer1_values': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1_values)),
                            'kmer2_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2_indices_0)),
                            'kmer2_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2_indices_1)),
                            'kmer2_values': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2_values)),
                            'score': tf.train.Feature(float_list=tf.train.FloatList(value=[score]))
                        }))
                writer.write(example.SerializeToString())
                if len(meta) == 0:
                    meta['input_shape'] = bow1_dense_shape

                columns = [idx1, idx2, score]
                fout.write(",".join([str(x) for x in columns]))
                fout.write("\n")
                counter+=1
                if counter % 1000 == 0:
                    print(".........", counter, "/", len(fixed_windows)*len(fixed_windows))

        np.save(outDir+"pretrain-meta.npy", meta)

    print("Writing fixed windows into csv...")
    with open(outDir+"fixed_windows.csv", "w") as fout2:
        for idx, win in enumerate(fixed_windows):
            fout2.write(str(idx) + "," + win + "\n")
    print("Fixed windows complete..")

window_cache = {}
def cached_bow_encoder(window):
    if window in window_cache:
        return window_cache[window]
    else:
        bow = Kmer_Utility.encodeKmerBagOfWords(window)
        window_cache[window] = bow
        return bow

def make_example(bow1, bow2, score):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'kmer1_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1[0])),
                'kmer1_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1[1])),
                'kmer1_values': tf.train.Feature(int64_list=tf.train.Int64List(value=bow1[2])),
                'kmer2_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2[0])),
                'kmer2_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2[1])),
                'kmer2_values': tf.train.Feature(int64_list=tf.train.Int64List(value=bow2[2])),
                'score': tf.train.Feature(float_list=tf.train.FloatList(value=[score]))
            }))
    return example

def each_worker(bow1, bow_win1, fixed):
    bow_win2 = cached_bow_encoder(fixed)
    bow2 = Kmer_Utility.sparseRepresentation(bow_win2)
    score = euclidean_distance(bow_win1, bow_win2)
    example = make_example(bow1, bow2, score)
    return example

def write_non_fixed_points_parallel(fixed_windows, all_windows):
    total = len(fixed_windows)*len(all_windows)
    counter = 0
    last_encoded_window, last_window = None, None
    examples = []



    print("Computing non-fixed points....")
    for window in all_windows:
        bow_win1 = Kmer_Utility.encodeKmerBagOfWords(window, last_encoded_window=last_encoded_window, last_window=last_window)
        last_encoded_window = bow_win1
        last_window = window
        bow1 = Kmer_Utility.sparseRepresentation(bow_win1)
        start_time = time.time()
        batch_examples = Parallel(n_jobs=1, verbose=0, backend="threading")(delayed(each_worker)(bow1, bow_win1, fixed) for fixed in fixed_windows)
        examples = examples + batch_examples
        print("--- %s seconds ---" % (time.time() - start_time))
        counter += len(fixed_windows)
        if counter%1000 == 0:
            print("......", counter, "/", total)



def write_non_fixed_points(fixed_windows, all_windows):
    total = len(fixed_windows)*len(all_windows)
    counter = 0
    last_encoded_window, last_window = None, None
    examples = []
    meta = {}

    print("Computing non-fixed points....")
    for window in all_windows:
        bow_win1 = Kmer_Utility.encodeKmerBagOfWords(window, last_encoded_window=last_encoded_window, last_window=last_window)
        last_encoded_window = bow_win1
        last_window = window
        bow1 = Kmer_Utility.sparseRepresentation(bow_win1)
        reset_time = time.time()
        for fixed in fixed_windows:
            bow_win2 = cached_bow_encoder(fixed)
            bow2 = Kmer_Utility.sparseRepresentation(bow_win2)
            score = euclidean_distance(bow_win1, bow_win2)
            example = make_example(bow1, bow2, score)
            examples.append(example)
            counter += 1
            if counter % 1000 == 0:
                print(".........", counter, "/", total, "--- %s seconds ---" % (time.time() - reset_time))
                reset_time = time.time()
            if len(meta) == 0:
                meta['input_shape'] = bow1[3]

    meta['total'] = len(examples)
    print("Test train splits..")
    train, test = train_test_split(examples, test_size=0.20)
    meta['train_size'] = len(train)
    meta['test_size'] = len(test)

    def write(split, filename):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for df in split:
                writer.write(df.SerializeToString())

    print("Write train split..")
    write(train, outDir+"/train.tfrecords")
    print("Write test split..")
    write(train, outDir+"/test.tfrecords")
    print("Writing meta")
    np.save(outDir+"/meta.npy", meta)
    print(meta)


if __name__=="__main__":
    genome = Kmer_Utility.readGenome(outDir+"sequence.fasta")
    windows = Kmer_Utility.slidingWindow(genome, WINDOW_LEN)
    fixed_windows = np.random.choice(windows, size=FIXED_POINTS)
    # write_fixed_points(fixed_windows)
    # write_non_fixed_points_parallel(fixed_windows, all_windows=windows)
    write_non_fixed_points(fixed_windows, all_windows=windows)





