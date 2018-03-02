import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import math
import copy


workDir = "run/"
SEGMENTS_LEN = 10000
WINDOW_LEN = 1000
nucMap = {'A':0, 'C':1, 'G':2, 'T':3}

kmerEncoding = {}
def encodeKmer(kmer):
    if kmer in kmerEncoding:
        return kmerEncoding[kmer]

    numeric = 0
    for idx, nt in enumerate(kmer):
        numeric += nucMap[nt] * 4**idx
    kmerEncoding[kmer] = numeric
    return numeric


def readGenome(fasta):
    lines = ""
    with open(fasta, "r") as f:
        for idx, line in enumerate(f):
            if idx > 0:
                lines += line.strip()
    return lines

def getSegments(genome, segmentLength, windowLen):
    genomeLength = len(genome)
    numSegments = math.ceil(float(genomeLength)/float(segmentLength))
    segments = []
    for i in range(numSegments):
        start = max(0, i*SEGMENTS_LEN - windowLen)
        end = min(start + SEGMENTS_LEN + 2*windowLen, genomeLength)
        segment = genome[start:end]
        segments.append((i, segment))
    return numSegments, segments

def slidingWindow(segment, winlength):
    windows = []
    for i in range(len(segment)-winlength+1):
        window = segment[i:i+winlength]
        windows.append(window)
    return windows


def stringifyLabels(segIds, numSegments):
    commaSep = ",".join(str(seg) for seg in segIds)
    return (commaSep + "$" + str(numSegments)).encode()

# actually turns to numpy array of number according mapping in nucMap[]
def stringifyFeature(feature):
    charArr = [nucMap[x] for x in list(feature)]
    return np.asarray(charArr)

def encodeKmerBagOfWords(window, last_window, last_encoded_window):
    k = 8

    # incremental -- Not sure if dictionary order is preserved
    if last_window is not None and last_encoded_window is not None:
        last_start_kmer = last_window[0:0+k]
        last_encoded_window[encodeKmer(last_start_kmer)] -= 1
        this_end_kmer = window[-k:]
        last_encoded_window[encodeKmer(this_end_kmer)] += 1
        return last_encoded_window

    # sliding kmer extraction
    b = np.zeros(4**k)
    for i in range(len(window) - k +1 ):
        kmer_num = encodeKmer( window[i:i+k] )
        b[kmer_num] = 1
    return b.flatten()

def encodeLabels(segIds, numSegments):
    label_indices = segIds
    label_values = np.ones(np.shape(segIds)[0], dtype=np.int8)
    label_dense_shape = np.array([numSegments, 1])
    return label_indices, label_values, label_dense_shape

def sparseRepresentation(arr):
    indices = np.where(arr)[0]
    values = np.ones(np.shape(indices)[0], dtype=np.int8)
    dense_shape = np.array([np.shape(arr)[0], 1])
    return indices, values, dense_shape

def writeTFRecord(df, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for row in df:
            label_indices_1 = np.zeros(len(row['label_indices']), dtype=np.int8).tolist()
            label_values = np.ones(len(row['label_indices']), dtype=np.int8).tolist()
            feature_indices_1 = np.zeros(len(row['feature_indices']), dtype=np.int8).tolist()
            feature_values = np.ones(len(row['feature_indices']), dtype=np.int8).tolist()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'feature_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=row['feature_indices'])),
                        'feature_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=feature_indices_1)),
                        'feature_values': tf.train.Feature(int64_list=tf.train.Int64List(value=feature_values)),
                        'label_indices_0': tf.train.Feature(int64_list=tf.train.Int64List(value=row['label_indices'])),
                        'label_indices_1': tf.train.Feature(int64_list=tf.train.Int64List(value=label_indices_1)),
                        'label_values': tf.train.Feature(int64_list=tf.train.Int64List(value=label_values))
                    }))
            writer.write(example.SerializeToString())

def writeMeta(meta):
    np.save(workDir+"meta.npy", meta)
    print(meta)


genome = readGenome(workDir+"sequence.fasta")
numSegments, segments = getSegments(genome, SEGMENTS_LEN, WINDOW_LEN) #numSegments is needed for output layer
allWindows = {}
for segment in segments:
    segId = segment[0]
    windows = slidingWindow(segment[1], WINDOW_LEN)
    for window in windows:
        if window in allWindows:
            allWindows[window].append(segId)
        else:
            allWindows[window] = [segId]
print("All windows computed...")

# Create Dataframe
print("Starting to encode...")
counter = 0
df = []
meta = {}
last_window, last_encoded_window = None, None
for key,values in allWindows.items():
    features = encodeKmerBagOfWords(key, last_window, last_encoded_window)
    last_window, last_encoded_window = key, copy.deepcopy(features)

    features_indices, features_values, features_dense_shape = sparseRepresentation(features)
    # print(features_indices, features_values, features_dense_shape)
    label_indices, label_values, label_dense_shape = encodeLabels(values, numSegments)
    # print(label_indices, label_values, label_dense_shape)
    df.append({"feature_indices":features_indices,
                    # "feature_values": features_values,
                    "label_indices": label_indices,
                    # "label_values": label_values
               })
    if len(meta) == 0:
        meta = {"feature_dense_shape": features_dense_shape,"label_dense_shape": label_dense_shape}
    counter += 1
    if counter %10000 == 0:
        print(str(counter) + " encoded out of " + str(len(allWindows)))
    # if counter == 100000:
    #     break

print("Test train split.....")
train, test = train_test_split(df, test_size=0.30)
# print(train.head(10))
# print(test.head(10))
print("Writing train to file...")
writeTFRecord(train, workDir+"train.tfrecords")
print("Writing test to file...")
writeTFRecord(test, workDir+"test.tfrecords")
writeMeta(meta)
exit(0)


a = "CAG"
charArr = [nucMap[x] for x in list(a)]

s = tf.one_hot(charArr, len(nucMap))
s = tf.reshape(s, [1, -1])

sess = tf.Session()
op = sess.run(s)
print(op)
print(encodeKmerBagOfWords(a))
print(sparseRepresentation(encodeKmerBagOfWords(a)))
print("sum", np.sum(np.array(op) - encodeKmerBagOfWords(a)))