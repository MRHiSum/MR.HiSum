import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf

# This code is created by referring to the code of https://github.com/google/youtube-8m

def parse_sequence_example(example_proto):
    context_features = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64)
    }
    sequence_features = {
        'rgb': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'audio': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }

    return tf.io.parse_sequence_example(example_proto, context_features, sequence_features)

def dequantize(features):
    return features * 4.0 / 255.0 - 2.0

def read_tfrecord(dataset_path, file_name, random_id):
    yt8m_path = os.path.join(dataset_path, f"frame/{file_name}.tfrecord")
    dataset = tf.data.TFRecordDataset(yt8m_path)
    parsed_dataset = dataset.map(parse_sequence_example)
    for example in parsed_dataset:
        if random_id == example[0]['id'].numpy().decode('utf-8'):
            label_indices = example[0]['labels'].values
            features = tf.cast(tf.io.decode_raw(example[1]['rgb'], tf.uint8), tf.float32).numpy()
            break
    
    return dequantize(features), label_indices.numpy().astype(int)