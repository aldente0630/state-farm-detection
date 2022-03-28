import argparse
import json
import logging
import os
from collections.abc import Iterable
from functools import partial
import tensorflow as tf
import tensorflow_addons as tfa
from albumentations import (
    Compose,
    ShiftScaleRotate,
)
from sagemaker_tensorflow import PipeModeDataset

logging.getLogger().setLevel(logging.INFO)


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError


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


def _parse_function(example_proto, n_classes=None, is_prediction=False):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/file_name": tf.io.FixedLenFeature([], tf.string),
    }
    if not is_prediction:
        features["image/label"] = (
            tf.io.FixedLenFeature([], tf.float32)
            if n_classes is None
            else tf.io.FixedLenFeature([n_classes], tf.float32)
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
    mode,
    channel,
    channel_name,
    image_size,
    transforms,
    batch_size,
    shuffle=True,
    buffer_size=10240,
    n_classes=None,
    is_prediction=False,
    normalize=True,
):
    if mode == "Pipe":
        raw_dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    else:
        tfrecord_names = tf.io.gfile.glob(os.path.join(channel, "*.tfrec"))
        raw_dataset = tf.data.TFRecordDataset(filenames=tfrecord_names)

    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        partial(_parse_function, n_classes=n_classes, is_prediction=is_prediction),
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


def sample_beta_dist(size, concentration_x=0.2, concentration_y=0.2):
    gamma_x_sample = tf.random.gamma(shape=[size], alpha=concentration_y)
    gamma_y_sample = tf.random.gamma(shape=[size], alpha=concentration_x)
    return gamma_x_sample / (gamma_x_sample + gamma_y_sample)


def get_model(
    img_size,
    fc_size,
    n_classes,
    initial_learning_rate,
    first_decay_steps,
    use_adamw,
    use_swa,
    label_smoothing,
    verbose=False,
):
    eff_net = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    eff_net.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(eff_net.output)
    x = tf.keras.layers.Dense(fc_size, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(fc_size, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(eff_net.input, output)

    if verbose:
        print(model.summary())

    # For tensorflow 2.5 or later, use tf.keras.optimizers.schedules.CosineDecayRestarts.
    lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate, first_decay_steps
    )
    optimizer = (
        tfa.optimizers.AdamW(lr_decayed_fn)
        if use_adamw
        else tfa.optimizers.RectifiedAdam(lr_decayed_fn)
    )
    if use_swa:
        optimizer = tfa.optimizers.SWA(optimizer, start_averaging=0, average_period=10)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["acc"],
    )

    return model


def main(args):
    train_steps_per_epoch = round(args.n_train_examples / args.batch_size)
    valid_steps_per_epoch = round(args.n_valid_examples / args.batch_size)

    train_transforms = Compose(
        [
            ShiftScaleRotate(
                rotate_limit=(-20, 20),
                scale_limit=(0.0, 0.2),
                shift_limit_x=(-0.0625, 0.0625),
                shift_limit_y=(-0.046875, 0.046875),
                p=1.0,
            ),
        ]
    )
    valid_transforms = train_transforms

    mode = args.input_data_config["train"]["TrainingInputMode"]
    one_train_dataset = load_tfrecord_dataset(
        mode,
        args.train,
        "train",
        args.img_size,
        train_transforms,
        args.batch_size,
        n_classes=args.n_classes,
        normalize=False,
    )

    if args.use_mixup:
        oth_train_dataset = load_tfrecord_dataset(
            mode,
            args.train,
            "train",
            args.img_size,
            train_transforms,
            args.batch_size,
            n_classes=args.n_classes,
            normalize=False,
        )
        zipped = tf.data.Dataset.zip((one_train_dataset, oth_train_dataset))
        train_dataset = zipped.map(
            lambda x, y: mixup_dataset(x, y, alpha=0.2),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        train_dataset = one_train_dataset

    mode = args.input_data_config["validation"]["TrainingInputMode"]
    valid_dataset = load_tfrecord_dataset(
        mode,
        args.validation,
        "validation",
        args.img_size,
        valid_transforms,
        args.batch_size,
        n_classes=args.n_classes,
        normalize=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "model.h5"),
            monitor="val_loss",
        ),
        tf.keras.callbacks.TensorBoard(
            args.tf_logs_path,
            histogram_freq=0,
            update_freq="epoch",
            profile_batch=0,
        ),
    ]

    model = get_model(
        args.img_size,
        args.fc_size,
        args.n_classes,
        args.initial_learning_rate,
        args.first_decay_steps,
        args.use_adamw,
        args.use_swa,
        args.label_smoothing,
    )

    model.fit(
        train_dataset,
        epochs=args.n_epochs,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=valid_steps_per_epoch,
        callbacks=callbacks,
    )

    if args.current_host == args.hosts[0]:
        version = "1"
        model.save(os.path.join(args.model_dir, version))

    logging.info("The model training task has been successfully completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--fc_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--first_decay_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--use_adamw",
        type=str_to_bool,
        default=False,
    )
    parser.add_argument(
        "--use_swa",
        type=str_to_bool,
        default=False,
    )
    parser.add_argument(
        "--use_mixup",
        type=str_to_bool,
        default=False,
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--n_train_examples",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--n_valid_examples",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--n_classes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--tf_logs_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
    )
    parser.add_argument(
        "--current_host", type=str, default=os.environ.get("SM_CURRENT_HOST")
    )
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS"))
    )
    parser.add_argument(
        "--input_data_config",
        type=json.loads,
        default=os.environ.get("SM_INPUT_DATA_CONFIG"),
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR")
    )

    args = parser.parse_args()
    main(args)
