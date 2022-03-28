import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_img_arr(img_path, img_size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_arr, axis=0)


def make_gradcam_heatmap(img_arr, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_arr)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap_arr = tf.squeeze(last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap_arr = tf.maximum(heatmap_arr, 0.0) / tf.math.reduce_max(heatmap_arr)
    return heatmap_arr.numpy(), pred_index


def make_gradcam_img(img_path, heatmap_arr, alpha=0.4, cam_path=None):
    img_arr = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(img_path)
    )
    heatmap_arr = np.uint8(255.0 * heatmap_arr)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap_arr = jet_colors[heatmap_arr]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap_arr)
    jet_heatmap = jet_heatmap.resize((img_arr.shape[1], img_arr.shape[0]))
    jet_heatmap_arr = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img_arr = jet_heatmap_arr * alpha + img_arr

    if cam_path is not None:
        superimposed_img = tf.keras.preprocessing.image.array_to_img(
            superimposed_img_arr
        )
        superimposed_img.save(cam_path)
    return superimposed_img_arr


def view_img(img_arrs, labels, n_samples, label_names=None):
    n_cols = 5
    n_rows = n_samples // n_cols if n_samples % n_cols == 0 else n_samples // n_cols + 1
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    for i in range(n_samples):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, xticks=[], yticks=[])
        ax.imshow(img_arrs[i])
        label = (
            labels[i].decode("utf-8")
            if label_names is None
            else label_names[f"c{str(labels[i])}"]
        )
        ax.set_title(f"Label: {label}")
        ax.axis("off")
