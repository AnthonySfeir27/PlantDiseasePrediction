#!/bin/bash

SOURCE="/home/anthony/Downloads/PlantVillage"
TARGET="/home/anthony/PycharmProjects/PlantDiseasePrediction/data/raw"

mkdir -p "$TARGET"

find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -exec cp -r {} "$TARGET/" \;

echo "Copied all class folders."
