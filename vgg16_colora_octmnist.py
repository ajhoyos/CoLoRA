from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.applications import VGG16
from keras.models import Sequential
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

LOGGER = logging.getLogger("vgg16_colora_octmnist")

DATASET_KEY = "octmnist"
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = {
    0: "Choroidal Neovascularization",
    1: "Diabetic Macular Edema",
    2: "Drusen",
    3: "Normal",
}
CLASS_ABBREVIATIONS = ["ChN", "DME", "Drusen", "Normal"]
DEFAULT_COLORA_BLOCKS = (
    "block2_conv",
    "block3_conv",
    "block4_conv",
    "block5_conv",
)
AUTOTUNE = tf.data.AUTOTUNE

VGG16_LAYER_NAMES = [
    "block1_conv1", "activa1_1", "block1_conv2", "activa1_2", "block1_pool",
    "block2_conv1", "activa2_1", "block2_conv2", "activa2_2", "block2_pool",
    "block3_conv1", "activa3_1", "block3_conv2", "activa3_2",
    "block3_conv3", "activa3_3", "block3_pool",
    "block4_conv1", "activa4_1", "block4_conv2", "activa4_2",
    "block4_conv3", "activa4_3", "block4_pool",
    "block5_conv1", "activa5_1", "block5_conv2", "activa5_2",
    "block5_conv3", "activa5_3", "block5_pool",
]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_mode: str = "distilled"
    distilled_path: Path = Path("./OCTMNISTv2_Distilled.npz")
    output_dir: Path = Path("./results")
    batch_size: int = 8
    epochs: int = 20
    seed: int = 42
    use_augmentation: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    backbone_weights: str | None = "imagenet"
    colora_blocks: tuple[str, ...] = DEFAULT_COLORA_BLOCKS


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_environment() -> None:
    LOGGER.info("TensorFlow %s | Keras %s", tf.__version__, keras.__version__)
    LOGGER.info("Visible GPUs: %s", tf.config.list_physical_devices("GPU"))


def medmnist_npz_path(
    dataset_key: str = DATASET_KEY,
    image_size: int = IMAGE_SIZE,
) -> Path:
    info = medmnist.INFO[dataset_key]
    suffix = f"_{image_size}" if image_size else ""
    url_key = f"url{suffix}"
    hash_key = f"MD5{suffix}"

    if hash_key not in info:
        hash_key = f"md5{suffix}"

    return Path(
        keras.utils.get_file(
            origin=info[url_key],
            file_hash=info[hash_key],
            hash_algorithm="md5",
        )
    )


def ensure_channel_axis(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return images


def class_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)

    if labels.ndim > 1 and labels.shape[-1] == NUM_CLASSES:
        labels = np.argmax(labels, axis=-1)

    labels = labels.reshape(-1)

    if not np.all(np.isfinite(labels)):
        raise ValueError("Labels contain non-finite values.")

    if not np.all(labels == np.floor(labels)):
        raise ValueError("Class labels must contain integer-valued class indices.")

    labels = labels.astype(np.int64)

    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError(
            f"Class labels must be in the range [0, {NUM_CLASSES - 1}]."
        )

    return labels


def one_hot(labels: np.ndarray) -> np.ndarray:
    return keras.utils.to_categorical(
        class_indices(labels),
        num_classes=NUM_CLASSES,
    )


def balanced_indices(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels = class_indices(labels)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    samples_per_class = int(class_counts.min())

    selected = [
        np.flatnonzero(labels == class_id)[:samples_per_class]
        for class_id in range(NUM_CLASSES)
    ]

    return np.sort(np.concatenate(selected)), samples_per_class


def load_octmnist_training_data(
    mode: str,
    distilled_path: Path,
    dataset_key: str = DATASET_KEY,
    image_size: int = IMAGE_SIZE,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, object],
]:
    mode = mode.lower()

    if mode not in {"full", "balanced", "distilled"}:
        raise ValueError("mode must be 'full', 'balanced', or 'distilled'.")

    data_path = medmnist_npz_path(dataset_key, image_size)

    with np.load(data_path) as data:
        val_images = np.array(data["val_images"])
        val_labels = class_indices(data["val_labels"])

        if mode in {"full", "balanced"}:
            train_images = np.array(data["train_images"])
            train_labels = class_indices(data["train_labels"])

    if mode == "balanced":
        selected_indices, _ = balanced_indices(train_labels)
        train_images = train_images[selected_indices]
        train_labels = train_labels[selected_indices]
    elif mode == "distilled":
        distilled_path = Path(distilled_path)

        if not distilled_path.exists():
            raise FileNotFoundError(
                f"Distilled dataset not found: {distilled_path.resolve()}"
            )

        with np.load(distilled_path) as distilled_data:
            missing_keys = {"X", "Y"} - set(distilled_data.files)
            if missing_keys:
                raise KeyError(
                    f"Distilled file is missing keys: {sorted(missing_keys)}"
                )

            train_images = np.array(distilled_data["X"])
            train_labels = class_indices(distilled_data["Y"])

    train_images = ensure_channel_axis(train_images)
    val_images = ensure_channel_axis(val_images)

    metadata = {
        "mode": mode,
        "train_samples": int(train_images.shape[0]),
        "validation_samples": int(val_images.shape[0]),
        "train_class_counts": np.bincount(
            train_labels,
            minlength=NUM_CLASSES,
        ).tolist(),
    }

    return (
        (train_images, one_hot(train_labels)),
        (val_images, one_hot(val_labels)),
        metadata,
    )


def load_octmnist_test_data(
    dataset_key: str = DATASET_KEY,
    image_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    data_path = medmnist_npz_path(dataset_key, image_size)

    with np.load(data_path) as data:
        test_images = ensure_channel_axis(np.array(data["test_images"]))
        test_labels = class_indices(data["test_labels"])

    return test_images, one_hot(test_labels)


def augment_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.flip_left_right(image)
    return image, label


def prepare_image(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32)
    return image, label


def build_training_datasets(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
    use_augmentation: bool,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(
        buffer_size=min(len(train_images), 10_000),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    if use_augmentation:
        train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.map(prepare_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.repeat().batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.map(prepare_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    steps_per_epoch = int(np.ceil(len(train_images) / batch_size))
    return train_ds, val_ds, steps_per_epoch


def build_test_dataset(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    dataset = dataset.map(prepare_image, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_extended_vgg16(img_size: int = IMAGE_SIZE) -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(64, 3, padding="same", name=VGG16_LAYER_NAMES[0])(inputs)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[1])(x)
    x = layers.Conv2D(64, 3, padding="same", name=VGG16_LAYER_NAMES[2])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[3])(x)
    x = layers.MaxPool2D(2, 2, name=VGG16_LAYER_NAMES[4])(x)

    x = layers.Conv2D(128, 3, padding="same", name=VGG16_LAYER_NAMES[5])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[6])(x)
    x = layers.Conv2D(128, 3, padding="same", name=VGG16_LAYER_NAMES[7])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[8])(x)
    x = layers.MaxPool2D(2, 2, name=VGG16_LAYER_NAMES[9])(x)

    x = layers.Conv2D(256, 3, padding="same", name=VGG16_LAYER_NAMES[10])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[11])(x)
    x = layers.Conv2D(256, 3, padding="same", name=VGG16_LAYER_NAMES[12])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[13])(x)
    x = layers.Conv2D(256, 3, padding="same", name=VGG16_LAYER_NAMES[14])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[15])(x)
    x = layers.MaxPool2D(2, 2, name=VGG16_LAYER_NAMES[16])(x)

    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[17])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[18])(x)
    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[19])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[20])(x)
    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[21])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[22])(x)
    x = layers.MaxPool2D(2, 2, name=VGG16_LAYER_NAMES[23])(x)

    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[24])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[25])(x)
    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[26])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[27])(x)
    x = layers.Conv2D(512, 3, padding="same", name=VGG16_LAYER_NAMES[28])(x)
    x = layers.Activation("relu", name=VGG16_LAYER_NAMES[29])(x)

    outputs = layers.MaxPool2D(2, 2, name=VGG16_LAYER_NAMES[30])(x)
    return keras.Model(inputs, outputs, name="vgg16")


def clone_vgg16_with_explicit_activations(
    pretrained_backbone: keras.Model,
) -> keras.Model:
    clone = build_extended_vgg16(img_size=IMAGE_SIZE)

    for layer in pretrained_backbone.layers:
        if isinstance(layer, layers.Conv2D):
            clone.get_layer(layer.name).set_weights(layer.get_weights())

    return clone


@keras.saving.register_keras_serializable(package="CoLoRA")
class Colora2D(keras.layers.Layer):
    def __init__(
        self,
        original_layer: keras.layers.Layer,
        trainable: bool = True,
        **kwargs,
    ):
        original_config = original_layer.get_config()
        name = kwargs.pop("name", f"{original_config['name']}_Colora")
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.original_layer = original_layer
        self.original_layer.trainable = False

        self.conv_1x1 = layers.Conv2D(
            filters=original_config["filters"],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=original_config["padding"],
            activation=original_config["activation"],
            use_bias=original_config["use_bias"],
            kernel_initializer="glorot_uniform",
        )

        self.conv_3x3 = layers.DepthwiseConv2D(
            kernel_size=original_config["kernel_size"],
            strides=original_config["strides"],
            padding=original_config["padding"],
            depth_multiplier=1,
            activation=original_config["activation"],
            use_bias=original_config["use_bias"],
            depthwise_initializer="zeros",
        )

    def build(self, input_shape) -> None:
        if not self.original_layer.built:
            self.original_layer.build(input_shape)

        if not self.conv_1x1.built:
            self.conv_1x1.build(input_shape)

        intermediate_shape = self.conv_1x1.compute_output_shape(input_shape)
        if not self.conv_3x3.built:
            self.conv_3x3.build(intermediate_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        original_output = self.original_layer(inputs)

        if training:
            update = self.conv_3x3(self.conv_1x1(inputs))
            return original_output + update

        return original_output

    def get_config(self) -> dict:
        config = super().get_config()
        config["original_layer"] = keras.saving.serialize_keras_object(
            self.original_layer
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "Colora2D":
        original_layer_config = config.pop("original_layer")
        original_layer = keras.saving.deserialize_keras_object(
            original_layer_config
        )
        return cls(original_layer=original_layer, **config)

    def merge_filters(self) -> None:
        original_weights = self.original_layer.get_weights()
        pointwise_weights = self.conv_1x1.get_weights()
        depthwise_weights = self.conv_3x3.get_weights()

        original_kernel = original_weights[0]
        original_bias = original_weights[1] if len(original_weights) > 1 else None
        pointwise_kernel = pointwise_weights[0]
        depthwise_kernel = depthwise_weights[0]
        depthwise_bias = depthwise_weights[1] if len(depthwise_weights) > 1 else None

        combined_kernel = tf.einsum(
            "abcd,efdg->efcd",
            pointwise_kernel,
            depthwise_kernel,
        ).numpy()

        merged_kernel = original_kernel + combined_kernel

        if original_bias is not None and depthwise_bias is not None:
            self.original_layer.set_weights(
                [merged_kernel, original_bias + depthwise_bias]
            )
        else:
            self.original_layer.set_weights([merged_kernel])

        reset_depthwise = np.zeros_like(depthwise_kernel)
        if depthwise_bias is not None:
            self.conv_3x3.set_weights(
                [reset_depthwise, np.zeros_like(depthwise_bias)]
            )
        else:
            self.conv_3x3.set_weights([reset_depthwise])

        reset_pointwise = keras.initializers.GlorotUniform()(
            self.conv_1x1.kernel.shape
        ).numpy()

        if len(pointwise_weights) > 1:
            self.conv_1x1.set_weights(
                [reset_pointwise, np.zeros_like(pointwise_weights[1])]
            )
        else:
            self.conv_1x1.set_weights([reset_pointwise])

    def set_training_mode(self, mode: str = "tuning") -> None:
        mode = mode.lower()

        if mode == "tuning":
            trainable_flags = (False, True, True)
        elif mode == "original":
            trainable_flags = (True, False, False)
        elif mode == "full":
            trainable_flags = (True, True, True)
        elif mode in {"freeze", "freaze"}:
            trainable_flags = (False, False, False)
        else:
            raise ValueError(
                "mode must be 'tuning', 'original', 'full', or 'freeze'."
            )

        (
            self.original_layer.trainable,
            self.conv_1x1.trainable,
            self.conv_3x3.trainable,
        ) = trainable_flags


def create_vgg16_classifier(weights: str | None = "imagenet") -> keras.Model:
    pretrained_backbone = VGG16(
        weights=weights,
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    backbone = clone_vgg16_with_explicit_activations(pretrained_backbone)
    backbone.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.Conv2D(
        128,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        name="compresser",
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", name="fc_1")(x)
    outputs = layers.Dense(NUM_CLASSES, name="fc_2")(x)

    model = keras.Model(inputs, outputs, name="vgg16_based")
    del pretrained_backbone
    return model


def add_colora_by_block(
    original_model: keras.Model,
    block_prefixes: Sequence[str],
) -> keras.Model:
    block_prefixes = tuple(block_prefixes)
    model = Sequential(name="vgg16_colora")
    model.add(keras.Input(shape=original_model.input_shape[1:]))

    for layer in original_model.layers[1:]:
        if layer.name == "vgg16":
            for backbone_layer in layer.layers:
                # The outer Sequential model already defines the input.
                if isinstance(backbone_layer, layers.InputLayer):
                    continue

                should_adapt = (
                    isinstance(backbone_layer, layers.Conv2D)
                    and any(
                        backbone_layer.name.startswith(prefix)
                        for prefix in block_prefixes
                    )
                )

                model.add(
                    Colora2D(backbone_layer)
                    if should_adapt
                    else backbone_layer
                )
        else:
            model.add(layer)

    return model


def set_colora_training(model: keras.Model, mode: str = "tuning") -> None:
    for layer in model.layers:
        if isinstance(layer, Colora2D):
            layer.set_training_mode(mode)


def merge_colora_filters(model: keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, Colora2D):
            layer.merge_filters()


def count_model_parameters(model: keras.Model) -> dict[str, int]:
    trainable = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": trainable,
    }


def collect_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    training: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    logits_batches = []
    label_batches = []

    for images, labels in dataset:
        logits_batches.append(model(images, training=training).numpy())
        label_batches.append(labels.numpy())

    return (
        np.concatenate(logits_batches, axis=0),
        np.concatenate(label_batches, axis=0),
    )


def experiment_tag(blocks: Sequence[str]) -> str:
    if not blocks:
        return "transfer_learning"

    block_numbers = []
    for block in blocks:
        match = re.search(r"block(\d+)", block)
        if match:
            block_numbers.append(match.group(1))

    if block_numbers:
        return "blocks_" + "_".join(block_numbers)

    return "_".join(blocks)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=json_default)


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["Epoch"], history["Train Loss"], label="Train")
    axes[0].plot(
        history["Epoch"],
        history["Validation Loss"],
        label="Validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history["Epoch"],
        history["Train Accuracy"],
        label="Train",
    )
    axes[1].plot(
        history["Epoch"],
        history["Validation Accuracy"],
        label="Validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_model(
    config: ExperimentConfig,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    steps_per_epoch: int,
) -> tuple[Path, pd.DataFrame, Path]:
    keras.utils.set_random_seed(config.seed)

    base_model = create_vgg16_classifier(weights=config.backbone_weights)
    model = add_colora_by_block(base_model, config.colora_blocks)
    del base_model
    gc.collect()

    set_colora_training(model, mode="tuning")

    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"],
        jit_compile=False,
    )

    parameter_counts = count_model_parameters(model)
    LOGGER.info(
        "Model parameters: total=%d, trainable=%d",
        parameter_counts["total_parameters"],
        parameter_counts["trainable_parameters"],
    )

    best_val_accuracy = -np.inf
    best_epoch = None
    best_weights = None
    history_rows = []

    for epoch in range(1, config.epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, config.epochs)

        fit_history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=1,
        )

        # CoLoRA is unmerged during training, so validation must include the branch.
        val_logits, val_labels = collect_predictions(
            model,
            val_dataset,
            training=True,
        )

        val_true = np.argmax(val_labels, axis=1)
        val_pred = np.argmax(val_logits, axis=1)
        val_accuracy = float(np.mean(val_true == val_pred))
        val_loss = float(
            keras.losses.BinaryCrossentropy(from_logits=True)(
                val_labels,
                val_logits,
            ).numpy()
        )

        train_loss = float(fit_history.history["loss"][-1])
        train_accuracy = float(fit_history.history["accuracy"][-1])

        history_rows.append(
            {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_accuracy,
            }
        )

        LOGGER.info(
            "train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f",
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_weights = [
                np.array(weight, copy=True)
                for weight in model.get_weights()
            ]
            LOGGER.info(
                "New best validation accuracy %.6f at epoch %d",
                best_val_accuracy,
                best_epoch,
            )

    if best_weights is None or best_epoch is None:
        raise RuntimeError("No validation checkpoint was stored.")

    model.set_weights(best_weights)
    merge_colora_filters(model)

    experiment_dir = (
        config.output_dir
        / f"{config.dataset_mode}_{experiment_tag(config.colora_blocks)}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_path = experiment_dir / "best_validation_model.keras"
    history_path = experiment_dir / "training_history.csv"
    curves_path = experiment_dir / "training_curves.png"

    model.save(str(model_path))

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)
    save_training_curves(history_df, curves_path)

    training_metadata = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        **parameter_counts,
    }
    save_json(training_metadata, experiment_dir / "training_metadata.json")

    LOGGER.info("Selected epoch: %d", best_epoch)
    LOGGER.info("Best validation accuracy: %.6f", best_val_accuracy)
    LOGGER.info("Saved model: %s", model_path)

    return model_path, history_df, experiment_dir


def evaluate_logits(
    logits: np.ndarray,
    labels_one_hot: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    y_true = np.argmax(labels_one_hot, axis=1)
    y_pred = np.argmax(probabilities, axis=1)

    test_loss = float(
        keras.losses.BinaryCrossentropy(from_logits=True)(
            labels_one_hot,
            logits,
        ).numpy()
    )
    accuracy = float(np.mean(y_true == y_pred))

    auc_per_class = [
        roc_auc_score(
            labels_one_hot[:, class_id],
            probabilities[:, class_id],
        )
        for class_id in range(NUM_CLASSES)
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )

    specificity = []
    total = matrix.sum()

    for class_id in range(NUM_CLASSES):
        tp = matrix[class_id, class_id]
        fn = matrix[class_id, :].sum() - tp
        fp = matrix[:, class_id].sum() - tp
        tn = total - tp - fn - fp
        specificity.append(tn / (tn + fp) if (tn + fp) else np.nan)

    summary = pd.Series(
        {
            "Test Loss": test_loss,
            "Accuracy": accuracy,
            "Macro AUC (OvR)": float(np.mean(auc_per_class)),
            "Macro Precision": float(np.mean(precision)),
            "Macro Recall": float(np.mean(recall)),
            "Macro F1": float(np.mean(f1)),
        },
        name="Value",
    )

    per_class = pd.DataFrame(
        {
            "Class": CLASS_ABBREVIATIONS,
            "AUC": auc_per_class,
            "Recall": recall,
            "Precision": precision,
            "Specificity": specificity,
            "F1": f1,
        }
    )

    return summary, per_class, matrix


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ABBREVIATIONS,
    )
    display.plot(values_format="d")
    plt.title("OCTMNIST held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_saved_model(
    model_path: Path,
    test_dataset: tf.data.Dataset,
    output_dir: Path,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    model = keras.models.load_model(
        str(model_path),
        custom_objects={"Colora2D": Colora2D},
    )

    logits, labels_one_hot = collect_predictions(
        model,
        test_dataset,
        training=False,
    )
    summary, class_metrics, matrix = evaluate_logits(logits, labels_one_hot)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_frame().to_csv(output_dir / "test_summary.csv")
    class_metrics.to_csv(output_dir / "test_class_metrics.csv", index=False)
    np.savetxt(
        output_dir / "test_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )
    save_confusion_matrix(
        matrix,
        output_dir / "test_confusion_matrix.png",
    )

    LOGGER.info("Test metrics:\n%s", summary.to_string())
    LOGGER.info("Per-class metrics:\n%s", class_metrics.to_string(index=False))

    return summary, class_metrics, matrix


def run_training(config: ExperimentConfig) -> tuple[Path, Path]:
    (train_images, train_labels), (val_images, val_labels), metadata = (
        load_octmnist_training_data(
            mode=config.dataset_mode,
            distilled_path=config.distilled_path,
        )
    )

    LOGGER.info(
        "Dataset mode=%s | train=%d | validation=%d | class_counts=%s",
        metadata["mode"],
        metadata["train_samples"],
        metadata["validation_samples"],
        metadata["train_class_counts"],
    )

    train_dataset, val_dataset, steps_per_epoch = build_training_datasets(
        train_images,
        train_labels,
        val_images,
        val_labels,
        batch_size=config.batch_size,
        use_augmentation=config.use_augmentation,
        seed=config.seed,
    )

    del train_images, train_labels, val_images, val_labels
    gc.collect()

    model_path, _, experiment_dir = train_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        steps_per_epoch=steps_per_epoch,
    )

    config_dict = asdict(config)
    config_dict["distilled_path"] = str(config.distilled_path)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["colora_blocks"] = list(config.colora_blocks)
    save_json(config_dict, experiment_dir / "experiment_config.json")
    save_json(metadata, experiment_dir / "dataset_metadata.json")

    return model_path, experiment_dir


def run_evaluation(
    model_path: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    # The held-out test split is loaded only for the final evaluation.
    test_images, test_labels = load_octmnist_test_data()
    test_dataset = build_test_dataset(
        test_images,
        test_labels,
        batch_size=batch_size,
    )

    evaluate_saved_model(
        model_path=model_path,
        test_dataset=test_dataset,
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate VGG16-CoLoRA on OCTMNIST."
    )
    parser.add_argument(
        "--run-mode",
        choices=("train-eval", "train", "eval"),
        default="train-eval",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("full", "balanced", "distilled"),
        default="distilled",
    )
    parser.add_argument(
        "--distilled-path",
        type=Path,
        default=Path("./OCTMNISTv2_Distilled.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--backbone-weights",
        default="imagenet",
        help="Keras VGG16 weights identifier. Use 'none' for random initialization.",
    )
    parser.add_argument(
        "--colora-blocks",
        nargs="*",
        default=list(DEFAULT_COLORA_BLOCKS),
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.run_mode == "eval" and args.model_path is None:
        raise ValueError("--model-path is required when --run-mode=eval.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_logging(args.verbose)
    log_environment()

    backbone_weights = (
        None
        if str(args.backbone_weights).lower() == "none"
        else args.backbone_weights
    )

    config = ExperimentConfig(
        dataset_mode=args.dataset_mode,
        distilled_path=args.distilled_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        use_augmentation=not args.no_augmentation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        backbone_weights=backbone_weights,
        colora_blocks=tuple(args.colora_blocks),
    )

    keras.utils.set_random_seed(config.seed)

    if args.run_mode in {"train-eval", "train"}:
        model_path, experiment_dir = run_training(config)
    else:
        model_path = args.model_path
        experiment_dir = model_path.parent

    if args.run_mode in {"train-eval", "eval"}:
        run_evaluation(
            model_path=model_path,
            output_dir=experiment_dir,
            batch_size=config.batch_size,
        )


if __name__ == "__main__":
    main()
