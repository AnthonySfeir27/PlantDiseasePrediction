# Plant Disease Prediction

A Streamlit GUI prototype for a CNN-based plant disease detection project.

## Current Stage

This version contains the working application interface only. The prediction service is currently a deterministic demo service, not the final TensorFlow model.

## Run the App

```bash
source .venv/bin/activate
streamlit run app.py
```

## Planned Dataset Classes

Use a small PlantVillage subset first:

- `Tomato___healthy`
- `Tomato___Early_blight`
- `Tomato___Late_blight`
- `Tomato___Leaf_Mold`

Place dataset folders under:

```text
data/raw/
```

## Project Structure

```text
PlantDiseasePrediction/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── models/
│   │   └── prediction_result.py
│   ├── services/
│   │   └── demo_prediction_service.py
│   └── ui/
│       ├── layout.py
│       ├── prediction_view.py
│       └── session_state.py
```
