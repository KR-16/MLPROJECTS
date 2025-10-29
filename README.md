End-to-End ML Project – Student Score Predictor

Overview
This repository contains an end-to-end machine learning application that predicts a student's math score based on demographics, lunch, test preparation, and reading/writing scores. It includes:
- A Flask web app with HTML templates for interactive prediction
- A trained model and preprocessing pipeline saved under `artifacts/`
- Modular source code for data ingestion, transformation, and model training under `src/`
- Notebooks for EDA and experimentation under `notebook/`

Tech Stack
- Python 3.8+
- Flask
- scikit-learn, pandas, numpy
- CatBoost (used during experimentation; see notebooks and `catboost_info/`)

Project Structure
```
.
├─ app.py                      # Flask entrypoint
├─ artifacts/                  # Saved model, preprocessor, datasets
│  ├─ model.pkl
│  ├─ preprocessor.pkl
│  ├─ train.csv | test.csv | raw.csv
├─ src/
│  ├─ components/              # Modular training pipeline pieces
│  │  ├─ data_ingestion.py
│  │  ├─ data_transformation.py
│  │  └─ model_trainer.py
│  ├─ pipeline/
│  │  ├─ predict_pipeline.py   # Loads preprocessor+model to serve predictions
│  │  └─ train_pipeline.py     # (placeholder for orchestration)
│  ├─ utils.py                 # Serialization, common helpers
│  ├─ logger.py | exception.py # Logging & custom exceptions
├─ templates/
│  ├─ index.html               # Landing page
│  └─ home.html                # Prediction form + results
├─ notebook/                   # EDA and model experiments
│  ├─ EDA.ipynb
│  └─ MODEL.ipynb
├─ requirements.txt
└─ README.md
```

Quickstart (Windows)
1) Clone and navigate
```bash
git clone <your-repo-url>
cd MLPROJECTS
```

2) Create virtual environment and install deps
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the Flask app
```bash
python app.py
```
The server starts on `http://0.0.0.0:5000` (or `http://127.0.0.1:5000`).

Using the Web App
- Navigate to `/` to see the landing page
- Go to `/predict_data` to use the form (`home.html`)
- Fill in:
  - gender
  - race_ethnicity
  - parental_level_of_education
  - lunch
  - test_preparation_course
  - reading_score (numeric)
  - writing_score (numeric)
- Submit to get the predicted math score displayed on the page

Programmatic Prediction
If you want to call the predictor in Python code:
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

data = CustomData(
    gender="male",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=70,
)

df = data.get_data_as_dataframe()
pred = PredictPipeline().predict(df)
print(pred[0])
```

Training and Artifacts
- The repository already includes a trained `model.pkl` and `preprocessor.pkl` in `artifacts/`, which the app uses at runtime.
- The modular training components live in `src/components/`:
  - `data_ingestion.py`: reads and splits data
  - `data_transformation.py`: builds preprocessing pipeline
  - `model_trainer.py`: trains and evaluates the model
- Notebooks in `notebook/` (`EDA.ipynb`, `MODEL.ipynb`) document data exploration and model experimentation. Use these as references for retraining.
- The `src/pipeline/train_pipeline.py` file is present as a placeholder; if you want a single-command retraining script, implement orchestration there using the components above, then run it with `python -m src.pipeline.train_pipeline`.

Datasets
- Example CSVs used for training/validation are stored in `artifacts/` (`raw.csv`, `train.csv`, `test.csv`). Replace these with your data or adapt the ingestion component to point to a new source.

Environment Variables
None are required for local usage by default. Logging and exception handling are configured in `src/logger.py` and `src/exception.py`.

Troubleshooting
- If the app cannot find `artifacts/model.pkl` or `artifacts/preprocessor.pkl`, ensure you have the files in place or retrain to regenerate them.
- Verify your Python version is 3.8+ and that `pip install -r requirements.txt` completed without errors.

License
Add your preferred license here (e.g., MIT) if distributing.
