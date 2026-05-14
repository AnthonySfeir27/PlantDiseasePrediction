"""Command-line training entry point."""

import argparse
from pathlib import Path

from src.config import DATASET_DIR
from src.training.trainer import train_model


def parse_arguments() -> argparse.Namespace:
    """Read command-line training options."""
    parser = argparse.ArgumentParser(description="Train the plant disease CNN model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs. Start with 5 to 8 for a fast student demo.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DATASET_DIR),
        help="Path to the dataset folder containing class subfolders.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the model using the configured dataset folder."""
    arguments = parse_arguments()
    train_model(dataset_dir=Path(arguments.dataset_dir), epochs=arguments.epochs)


if __name__ == "__main__":
    main()
