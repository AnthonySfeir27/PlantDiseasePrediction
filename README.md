# Plant Disease Prediction

A Streamlit application for tomato leaf disease classification using CNN transfer learning.

The GUI runs immediately using a clearly marked demo fallback. After training, it automatically switches to the saved TensorFlow model.

## Course Material Applied

- CNN image classification
- Transfer learning with MobileNetV2
- Data augmentation
- Train/validation split
- Confidence-based prediction
- Training history charts
- Streamlit interface for model demonstration

## Required Dataset Structure

Download the PlantVillage dataset and copy only these folders into `data/raw/`:

```text
data/raw/
├── Tomato___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
└── Tomato___Leaf_Mold/
```

Do not train on the full dataset first. These four classes are enough for a clean demo and faster training.

## Install Dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The GUI

```bash
streamlit run app.py
```

## Check Dataset Before Training

```bash
python scripts/check_dataset.py
```

## Train The Model

Fast demo training:

```bash
python train.py --epochs 5
```

Better training if the laptop handles it:

```bash
python train.py --epochs 8
```

Training creates:

```text
models/plant_disease_model.keras
models/class_names.json
models/training_history.json
models/accuracy_plot.png
models/loss_plot.png
```

The `.keras` model file is ignored by Git because it can be large. Keep it locally for the demo.

## Project Structure

```text
PlantDiseasePrediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── scripts/
│   └── check_dataset.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── config.py
│   ├── models/
│   │   ├── model_status.py
│   │   └── prediction_result.py
│   ├── services/
│   │   ├── demo_prediction_service.py
│   │   └── prediction_service.py
│   ├── training/
│   │   ├── data_loader.py
│   │   ├── dataset_validator.py
│   │   ├── model_builder.py
│   │   └── trainer.py
│   ├── ui/
│   │   ├── layout.py
│   │   ├── prediction_view.py
│   │   └── session_state.py
│   └── utils/
│       └── file_io.py
```

## Git Commands After Copying These Files

```bash
git add .
git commit -m "Add CNN training and prediction pipeline"
git push
```

## Latest UI additions

The app now shows:

- the number of trained classes loaded from `models/class_names.json`
- top 3 predictions for each image
- training/validation accuracy metrics from `models/training_history.json`
- saved accuracy and loss charts when available

This is useful when the top prediction has low confidence, because the second and third model choices explain what the model was considering.
