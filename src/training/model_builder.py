"""CNN transfer-learning model construction."""

import tensorflow as tf

from src.config import IMAGE_SIZE


def build_transfer_learning_model(class_count: int) -> tf.keras.Model:
    """Build a MobileNetV2-based image classifier.

    The MobileNetV2 base acts as a pretrained feature extractor.
    The custom top layers learn the tomato disease classes.
    """
    if class_count < 2:
        raise ValueError("At least two classes are required for classification.")

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="disease_classifier")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
