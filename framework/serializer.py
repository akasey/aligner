from framework.common import *

class DenseDataType:
    def __init__(self, key):
        self.key = key
        self.__create_constraints_meta()

    def __create_constraints_meta(self):
        constraints = {'dtype': tf.uint8}
        self.set_constraints(constraints)

    def get_constraints(self):
        return self.constraints

    def set_constraints(self, dictionary):
        self.constraints = dictionary

    def get_serializable_features(self, data):
        data = [data.astype(self.constraints['dtype'].as_numpy_dtype).tostring()]
        features = {
            str(self.key): tf.train.Feature(bytes_list=tf.train.BytesList(value=data))
        }
        return features

    def get_deserializable_features(self):
        return {
            str(self.key): tf.FixedLenFeature([], tf.string)
        }

    def get_tensor(self, parsed_example):
        data = tf.decode_raw(parsed_example[self.key], self.constraints['dtype'])
        return data

class SparseDataType:
    def __init__(self, key):
        self.key = key
        self.constraints = None

    def __create_constraints_meta(self, data):
        constraints = {
            'dtype_0': tf.uint16,
            'shape': np.array([np.shape(data)[0], 1]),
            'dtype_values': tf.uint8
        }
        self.set_constraints(constraints)

    def get_constraints(self):
        return self.constraints

    def set_constraints(self, dictionary):
        self.constraints = dictionary

    def get_serializable_features(self, data):
        if self.constraints is None:
            self.__create_constraints_meta(data)
        indices0, indices1, values, dense_shape = sparseRepresentation(data)
        indices0_str = [indices0.astype(self.constraints['dtype_0'].as_numpy_dtype).tostring()]
        values_str = [values.astype(self.constraints['dtype_values'].as_numpy_dtype).tostring()]

        features = {
            self.key+"_0": tf.train.Feature(bytes_list=tf.train.BytesList(value=indices0_str)),
            self.key+"_v": tf.train.Feature(bytes_list=tf.train.BytesList(value=values_str))
        }
        return features

    def get_deserializable_features(self):
        return {
            self.key + "_0": tf.FixedLenFeature([], tf.string),
            self.key + "_v": tf.FixedLenFeature([], tf.string),
        }

    def get_tensor(self, parsed_example):
        Y_0 = tf.cast(tf.decode_raw(parsed_example[self.key+'_0'], self.constraints['dtype_0']), dtype=tf.int64)
        Y_1 = tf.zeros(tf.shape(Y_0), dtype=tf.int64)
        Y_v = tf.decode_raw(parsed_example[self.key+'_v'], self.constraints['dtype_values'])
        Y_indices = tf.stack([Y_0,Y_1], axis=1)
        Y = tf.SparseTensor(indices=Y_indices, values=Y_v, dense_shape=self.constraints['shape'])
        Y = tf.sparse_tensor_to_dense(Y)
        Y = tf.squeeze(Y)
        return Y



class Serializer:
    def __init__(self, input):
        self.meta = None
        self.impl = ['dense', 'sparse']
        self.data_types = None

        if type(input) is dict:
            self.__init_dict(input)
        elif type(input) is str:
            self.__init_filename(input)
        else:
            raise ValueError("input not dictionary or filename")

    def __init_dict(self, dictionary):
        if self.__validate_meta(dictionary):
            self.meta = dictionary
            self.data_types = {}
            for k,v in self.meta.items():
                self.data_types[k] = self._get_data_type(k, v)
            return
        raise ValueError("Possible values in meta_dictionary are " + str(self.impl))

    def _get_data_type(self,key, value):
        if value == "dense":
            return DenseDataType(key)
        elif value == "sparse":
            return SparseDataType(key)
        return None

    def __validate_meta(self, meta):
        for key, value in meta.items():
            if value not in self.impl:
                return False
        return True

    def __init_filename(self, filename):
        metaTemp = np.load(filename)
        all_meta = {}
        for k in metaTemp.item():
            all_meta[k] = metaTemp.item().get(k)

        self.data_types = {}
        for k,v in all_meta['meta'].items():
            self.data_types[k] = self._get_data_type(k, v)
            self.data_types[k].set_constraints(all_meta['individual'][k])

    def make_serializable(self, **kwargs):
        features = {}
        for key in kwargs:
            if key not in self.data_types:
                raise ValueError("Undeclared key: %s" % (key))
            each_feature = self.data_types[key].get_serializable_features(kwargs[key])
            features.update(each_feature)
        each_row = tf.train.Example(
            features=tf.train.Features(
                feature=features
            ))
        return each_row.SerializeToString()

    def deserialization_feature_list(self):
        feature_list = {}
        for k,v in self.data_types.items():
            features = v.get_deserializable_features()
            feature_list.update(features)
        return feature_list

    def deserialize(self, deserializedThing):
        parsed_thing = tf.parse_single_example(deserializedThing, features=self.deserialization_feature_list())
        tensor_dict = {}
        for k,v in self.data_types.items():
            tensor_dict[k] = v.get_tensor(parsed_thing)
        return tensor_dict

    def save_meta(self, filename):
        cumulative_meta = {
            'meta': self.meta,
            'individual': {}
        }
        for k, v in self.data_types.items():
            cumulative_meta['individual'][k] = v.get_constraints()
        np.save(filename, cumulative_meta)


def test_write():
    writer = tf.python_io.TFRecordWriter("serialization_test.tfrecords")
    obj = Serializer({'X': 'sparse', 'Y': 'dense', 'Z': 'sparse'})
    serializable_features = obj.make_serializable(X=np.array([1,0,0,1,0,1,1]), Y=np.array([9,8,7,6,5,4,3,2,1]), Z=np.array([0,0,0,1,0,0,0,0]))
    writer.write(serializable_features)
    writer.close()
    print(serializable_features)
    obj.save_meta("serialization_test.npy")

def test_read():
    obj = Serializer("serialization_test.npy")
    next = tf.data.TFRecordDataset("serialization_test.tfrecords").map(obj.deserialize).make_one_shot_iterator().get_next()
    sess = tf.Session()
    restored_data = sess.run([next])
    print(restored_data)

if __name__=="__main__":
    test_write()
    test_read()
