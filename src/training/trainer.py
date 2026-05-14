"""Training orchestration for the plant disease classifier."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import CLASS_NAMES_PATH, MODEL_PATH, TRAINING_HISTORY_PATH, MODEL_DIR
from src.training.data_loader import load_training_and_validation_datasets
from src.training.dataset_validator import get_dataset_summary, validate_dataset
from src.training.model_builder import build_transfer_learning_model
from src.utils.file_io import write_json


def train_model(dataset_dir: Path, epochs: int = 8) -> None:
    """Train and save the CNN transfer-learning model."""
    validate_dataset(dataset_dir)
    train_dataset, validation_dataset, class_names = load_training_and_validation_datasets(dataset_dir)
    model = build_transfer_learning_model(class_count=len(class_names))

    callbacks = [
        # Keeps the best model from validation accuracy, useful for small student datasets.
        __create_model_checkpoint(),
        __create_early_stopping(),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    history_data = {
        "epochs_requested": epochs,
        "class_names": class_names,
        "dataset_summary": get_dataset_summary(dataset_dir),
        "history": history.history,
    }
    write_json(CLASS_NAMES_PATH, class_names)
    write_json(TRAINING_HISTORY_PATH, history_data)
    __save_training_plot(history.history)

    print(f"\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Classes saved to: {CLASS_NAMES_PATH}")
    print(f"History saved to: {TRAINING_HISTORY_PATH}")


def __create_model_checkpoint():
    """Create a callback that stores the best validation model."""
    from tensorflow.keras.callbacks import ModelCheckpoint

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )


def __create_early_stopping():
    """Create a callback that stops when validation accuracy stops improving."""
    from tensorflow.keras.callbacks import EarlyStopping

    return EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
    )


def __save_training_plot(history: dict[str, list[float]]) -> None:
    """Save accuracy/loss curves for the report and presentation."""
    if "accuracy" not in history or "val_accuracy" not in history:
        return

    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure()
    plt.plot(epochs, history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "accuracy_plot.png")
    plt.close()

    if "loss" in history and "val_loss" in history:
        plt.figure()
        plt.plot(epochs, history["loss"], label="Training Loss")
        plt.plot(epochs, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "loss_plot.png")
        plt.close()
