"""TensorFlow dataset loading functions."""

from pathlib import Path

import tensorflow as tf

from src.config import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED, VALIDATION_SPLIT


def load_training_and_validation_datasets(dataset_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load image folders into train and validation datasets."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_dataset.class_names
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)

    return train_dataset, validation_dataset, class_names
