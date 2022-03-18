import os
import re
import tensorflow as tf
from collections.abc import Iterable
from functools import partial
from tqdm.auto import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image_string, file_name, label=None):
    feature = {
        "image/encoded": _bytes_feature([image_string]),
        "image/file_name": _bytes_feature([file_name]),
    }
    if label is not None:
        if isinstance(label, Iterable):
            feature["image/label"] = _float_feature(label)
        else:
            feature["image/label"] = _int64_feature([label])
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def dump_tfrecord(examples, output_path, num_classes=None, is_prediction=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tf.io.TFRecordWriter(output_path) as writer:
        for example in tqdm(examples):
            if is_prediction:
                img_path, file_name = example
                label = None
            else:
                img_path, label, file_name = example
                label = int(re.search("[0-9]+", label, re.IGNORECASE).group())
                if num_classes is not None:
                    label = tf.keras.utils.to_categorical(
                        label, num_classes=num_classes, dtype="float32"
                    )

            image_string = open(img_path, "rb").read()
            file_name = file_name.encode("utf-8")

            writer.write(
                serialize_example(
                    image_string=image_string,
                    file_name=file_name,
                    label=label,
                )
            )


def _parse_function(example_proto, num_classes=None, is_prediction=False):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/file_name": tf.io.FixedLenFeature([], tf.string),
    }
    if not is_prediction:
        features["image/label"] = (
            tf.io.FixedLenFeature([], tf.float32)
            if num_classes is None
            else tf.io.FixedLenFeature([num_classes], tf.float32)
        )

    example = tf.io.parse_single_example(example_proto, features)
    img_arr = tf.image.decode_jpeg(example["image/encoded"], channels=3)

    if is_prediction:
        return img_arr, None
    return (
        img_arr,
        example["image/label"],
    )


def identity_function(image=None):
    return {"image": image}


def preprocess_image(image_array, label, image_size, normalize, transforms):
    def augment_function(img_arr, img_size):
        img_size = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
        data = {"image": img_arr}
        img_arr = transforms(**data)["image"]
        img_arr = tf.cast(img_arr / 255.0 if normalize else img_arr, tf.float32)
        img_arr = tf.image.resize(img_arr, size=[img_size[0], img_size[1]])
        return img_arr

    image_array = tf.numpy_function(
        func=augment_function, inp=[image_array, image_size], Tout=tf.float32
    )
    if label is None:
        return image_array
    return image_array, label


def load_tfrecord_dataset(
    tfrecord_names,
    image_size,
    transforms,
    batch_size,
    shuffle=True,
    buffer_size=10240,
    num_classes=None,
    is_prediction=False,
    normalize=True,
):
    raw_dataset = tf.data.TFRecordDataset(filenames=tfrecord_names).repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        partial(_parse_function, num_classes=num_classes, is_prediction=is_prediction),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.map(
        partial(
            preprocess_image,
            image_size=image_size,
            normalize=normalize,
            transforms=transforms,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    return dataset


def sample_beta_dist(size, concentration_x=0.2, concentration_y=0.2):
    gamma_x_sample = tf.random.gamma(shape=[size], alpha=concentration_y)
    gamma_y_sample = tf.random.gamma(shape=[size], alpha=concentration_x)
    return gamma_x_sample / (gamma_x_sample + gamma_y_sample)


def mixup_dataset(dataset1, dataset2, alpha=0.2):
    x1, y1 = dataset1
    x2, y2 = dataset2
    batch_size = tf.shape(x1)[0]

    lam = sample_beta_dist(batch_size, alpha, alpha)
    lam_x = tf.reshape(lam, (batch_size, 1, 1, 1))
    lam_y = tf.reshape(lam, (batch_size, 1))

    x = x1 * lam_x + x2 * (1.0 - lam_x)
    y = y1 * lam_y + y2 * (1.0 - lam_y)
    return x, y
