import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

MATCH = 1
MISMATCH = -1
GAP = -1

K=8
FIXED_POINTS = 2
OUTDIR="fixed-2"

nucMap = {'A':0, 'C':1, 'G':2, 'T':3}

def createDirIfNeeded():
    import os
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

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

def all_kmers(k):
    alphabets = ['A','C','G','T']
    N = len(alphabets)
    kmers = []
    for i in range(4**k):
        if i==0:
            word = [0] * k
        else:
            carryon = 1
            for i in reversed(range(k)):
                digit = word[i] + carryon
                word[i] = digit % N
                carryon = digit // N
                if carryon == 0:
                    break
        kmers.append("".join([alphabets[int(i)] for i in word]))
    return kmers

def encodeKmer(kmer):
    numeric = 0
    for idx, nt in enumerate(kmer):
        numeric += nucMap[nt] * 4**idx
    return numeric

def vectorize(kmer):
    b = np.zeros(4 ** len(kmer))
    idx = encodeKmer(kmer)
    b[idx] = 1
    return b

def sparseRepresentation(arr):
    indices0 = np.where(arr)[0]
    indices1 = np.zeros(np.shape(indices0)[0], dtype=np.int8)
    values = np.ones(np.shape(indices0)[0], dtype=np.int8)
    dense_shape = np.array([np.shape(arr)[0], 1])
    return indices0, indices1, values, dense_shape

def writeTFRecord(df, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for row in df:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'kmer1_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer1_indices_0'])),
                        'kmer1_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer1_indices_1'])),
                        'kmer1_values': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer1_values'])),
                        'kmer2_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer2_indices_0'])),
                        'kmer2_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer2_indices_1'])),
                        'kmer2_values': tf.train.Feature(int64_list=tf.train.Int64List(value=row['kmer2_values'])),
                        'score': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['score']]))
                    }))
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    createDirIfNeeded()

    kmers = all_kmers(K)
    print("Writing kmers tsv")
    with open(OUTDIR+"/kmers.tsv", "w") as fout:
        for kmer in kmers:
            fout.write(kmer+"\n")
        fout.close()

    random.shuffle(kmers)
    fixed_kmers = [kmers[i] for i in range(FIXED_POINTS)]
    print("Fixed points", fixed_kmers)

    df = []
    count = 0
    total = len(kmers) * FIXED_POINTS
    meta = {}
    for fixed in fixed_kmers:
        fixed_indices = vectorize(fixed)
        kmer1_indices_0, kmer1_indices_1, kmer1_values, kmer1_shape = sparseRepresentation(fixed_indices)
        for kmer in kmers:
            kmer_indices = vectorize(kmer)
            kmer2_indices_0, kmer2_indices_1, kmer2_values, kmer2_shape = sparseRepresentation(kmer_indices)
            score = needleman_wunsch(kmer, fixed)

            row = {
                'kmer1_indices_0': kmer1_indices_0,
                'kmer1_indices_1': kmer1_indices_1,
                'kmer1_values': kmer1_values,
                'kmer2_indices_0': kmer2_indices_0,
                'kmer2_indices_1': kmer2_indices_1,
                'kmer2_values': kmer2_values,
                'score': int(score)
            }
            df.append(row)
            count += 1

            if (count+1) % 10000 == 0:
                print(count, '/', total)
        if len(meta) == 0:
            meta = {"input_shape": kmer1_shape, "total": total}

    print("Test-Train split")
    train, test = train_test_split(df, test_size=0.20)
    print("Writing train to file")
    writeTFRecord(train, OUTDIR+"/train.tfrecords")
    print("Writing test to file")
    writeTFRecord(test, OUTDIR+"/test.tfrecords")
    print("Write meta")
    np.save(OUTDIR+"/meta.npy", meta)
    print("meta: ", meta)
    print("Completed")


