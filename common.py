import numpy as np
from tensorflow.python.client import device_lib

__HAS_GPU_RESULT = None
def has_gpu() -> bool:
    """Check if TensorFlow can access GPU.

    The test is based on
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/test.py
    ...but we are interested only in CUDA GPU devices.

    Returns:
        True, if TF can access the GPU
    """
    # pylint: disable=global-statement
    global __HAS_GPU_RESULT
    # pylint: enable=global-statement
    if __HAS_GPU_RESULT is None:
        __HAS_GPU_RESULT = any((x.device_type == 'GPU')
                               for x in device_lib.list_local_devices())
    return __HAS_GPU_RESULT


def sigmoid(x):
    x=np.array(x,dtype=np.float128)
    values = 1 / (1 + np.exp(-x))
    return values.astype(int)
