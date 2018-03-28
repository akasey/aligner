import argparse
import tensorflow as tf
import numpy as np
import logging
import train


logging.getLogger().setLevel(logging.INFO)
def getInputFn():
    # loader = Loader(FLAGS.data_dir, batch_size=None)
    def input_fn():
        datapoints = []
        for idx in np.arange(65536):
            x = np.zeros([65536, ], dtype=np.float32)
            x[idx] = 1
            datapoints.append(x)

        # dataset = tf.data.Dataset.from_tensor_slices({
        #     "x1": np.array(datapoints)
        # }).batch(5).repeat(False).make_one_shot_iterator().get_next()
        # return dataset

        return tf.estimator.inputs.numpy_input_fn(
            x={"x1": np.array(datapoints)},
            batch_size=512,
            num_epochs=1,
            shuffle=False
        )
    return input_fn()

def read_kmers():
    dictionary = {}
    with open(FLAGS.data_dir+"/kmers.tsv") as fin:
        for idx,line in enumerate(fin.readlines()):
            dictionary[idx] = line
    return dictionary

def main():
    dictionary = read_kmers()
    hparams = train.restore_model_params(flags=FLAGS)
    assert hparams is not None
    estimator, train_spec, eval_spec = train.create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=hparams.model_dir,
            save_checkpoints_steps=500,
            keep_checkpoint_max=2,
            save_summary_steps=100),
        model_params=hparams)

    predictions = estimator.predict(input_fn=getInputFn())
    embedding = np.zeros([65536, hparams.layers[-1]])
    for prediction in predictions:
        ip = prediction['features']
        one_idx = np.argwhere(ip).flatten()
        embedding[one_idx,: ] = prediction['logits']

    similarity = np.matmul(embedding, embedding.T)
    top_k = 10
    for i in np.random.randint(0,65536, 15):
        nearest = (-similarity[i, :]).argsort()[1:top_k + 1]
        print(dictionary[i], "close to", [dictionary[i] for i in nearest])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="fixed-2/",
        help="Where is input data dir? use data_generation.py to create one")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="runtime_junk/",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()

    main()
    # model_fn(None, None, None, None)