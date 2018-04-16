import logging
import tensorflow as tf
import sys
import numpy as np

def make_logger(name):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def make_session(device, memory_fraction=0.33):
    if "gpu" in device:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    else:
        return tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


def sparseRepresentation(arr):
    indices0 = np.where(arr)[0]
    indices1 = np.zeros(np.shape(indices0)[0], dtype=np.int8)
    values = arr[indices0]
    dense_shape = np.array([np.shape(arr)[0], 1])
    return indices0, indices1, values, dense_shape