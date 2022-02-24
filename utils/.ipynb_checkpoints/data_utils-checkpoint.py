import os
import tensorflow as tf
from collections.abc import Iterable
from functools import partial
from tqdm.auto import tqdm


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image_string, file_name, label=None):
    feature = {
        "image/encoded": _bytes_feature(image_string),
        "image/file_name": _bytes_feature(file_name),
    }
    if label is not None:
        feature["image/label"] = _bytes_feature(label)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dump_tfrecord(examples, output_path, is_prediction=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tf.io.TFRecordWriter(output_path) as writer:
        for example in tqdm(examples):
            if is_prediction:
                img_path, file_name = example
                label = None
            else:
                img_path, label, file_name = example
                label = label.encode("utf-8")

            image_string = open(img_path, "rb").read()
            file_name = file_name.encode("utf-8")

            writer.write(
                serialize_example(
                    image_string=image_string,
                    file_name=file_name,
                    label=label,
                )
            )


def _parse_function(example_proto, is_prediction=False):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/file_name": tf.io.FixedLenFeature([], tf.string),
    }
    if not is_prediction:
        features["image/label"] = tf.io.FixedLenFeature([], tf.string)

    example = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(example["image/encoded"], channels=3)

    if is_prediction:
        return image
    return (
        image,
        example["image/label"],
    )


def preprocess_image(image, label, image_size, transforms):
    def augment_function(img, img_size):
        data = {"image": img}
        aug_image = transforms(**data)["image"]
        aug_image = tf.cast(aug_image / 255.0, tf.float32)
        img_size = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
        aug_image = tf.image.resize(aug_image, size=[img_size[0], img_size[1]])
        return aug_image

    augmented_image = tf.numpy_function(
        func=augment_function, inp=[image, image_size], Tout=tf.float32
    )
    return augmented_image, label


def load_tfrecord_dataset(
    tfrecord_names,
    image_size,
    transforms,
    batch_size,
    shuffle=True,
    buffer_size=10240,
):
    raw_dataset = tf.data.TFRecordDataset(filenames=tfrecord_names).repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        partial(preprocess_image, image_size=image_size, transforms=transforms),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    return dataset
