# Churn Prediction with Full MLOps Pipeline

An end-to-end customer churn prediction project built with FastAPI, XGBoost, SMOTE, MLflow, and a lightweight browser frontend. The project trains a churn classifier on the Telco Customer Churn dataset, tracks experiments with MLflow, saves the trained model locally, and serves real-time predictions through an API and web form.

## What This Project Does

- Loads and preprocesses the Telco Customer Churn dataset.
- Cleans `TotalCharges` and removes the non-predictive `customerID` column.
- Encodes categorical columns for model training.
- Splits data with stratification to preserve churn distribution.
- Handles class imbalance using SMOTE.
- Trains an XGBoost classifier.
- Tracks metrics, parameters, and model artifacts with MLflow.
- Saves the trained model to `models/churn_model.pkl`.
- Serves predictions through FastAPI.
- Provides a simple frontend at `/` and Swagger docs at `/docs`.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data | pandas |
| Modeling | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn SMOTE |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Model Persistence | joblib |
| Frontend | HTML, CSS, JavaScript served by FastAPI |

## Repository Structure

```text
churn-mlops/
├── data/                         # Local dataset files, ignored by git
├── models/                       # Trained model artifacts, ignored by git
├── notebooks/                    # Optional notebooks
├── src/
│   ├── api.py                    # FastAPI app, prediction endpoint, and frontend
│   ├── predict.py                # API client smoke test
│   └── train.py                  # Preprocessing, training, MLflow tracking, model saving
├── .gitignore
├── README.md
└── requirements.txt
```

GitHub intentionally tracks only source and documentation. Large or generated files such as datasets, trained models, MLflow runs, virtual environments, and local databases are ignored.

## Dataset

This project uses the Kaggle Telco Customer Churn dataset:

```text
blastchar/telco-customer-churn
```

Expected local file:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The training script also checks for this shorter optional path first:

```text
data/telco.csv
```

Because `data/` is ignored by git, anyone cloning this repo must download the dataset locally before training.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Download Dataset

Install and configure the Kaggle CLI if needed:

```bash
pip install kaggle
```

Then download the dataset into `data/`:

```bash
mkdir -p data
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
```

## Train the Model

Run:

```bash
python src/train.py
```

The training script will:

1. Load the Telco dataset.
2. Drop `customerID`.
3. Convert `TotalCharges` to numeric.
4. Fill missing `TotalCharges` values with the median.
5. Encode `Churn` as `1` for `Yes` and `0` for `No`.
6. Label-encode categorical columns.
7. Create a stratified train/test split.
8. Balance the training split with SMOTE.
9. Train an XGBoost model.
10. Log parameters, metrics, and model artifacts to MLflow.
11. Save the trained model to `models/churn_model.pkl`.

Example output:

```text
Train shape: (5634, 19), Test shape: (1409, 19)
Train churn rate: 0.265, Test churn rate: 0.265
Accuracy: 0.766 | F1: 0.590 | AUC: 0.834
Model saved to models/churn_model.pkl
```

## MLflow Experiment Tracking

Start the MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Experiment name:

```text
churn-prediction
```

Run name:

```text
xgboost-baseline
```

Logged items include:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `resampling`
- `smote_random_state`
- `accuracy`
- `f1_score`
- `auc`
- training rows before and after SMOTE
- churn count before and after SMOTE
- model artifact

## Run the API and Frontend

Make sure `models/churn_model.pkl` exists. If not, run training first.

Start FastAPI:

```bash
uvicorn src.api:app --reload
```

Open the frontend:

```text
http://127.0.0.1:8000/
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Prediction API

Endpoint:

```http
POST /predict
```

Minimal request body:

```json
{
  "tenure": 12,
  "MonthlyCharges": 65.0,
  "TotalCharges": 780.0,
  "Contract": 0,
  "PaymentMethod": 2,
  "InternetService": 1
}
```

Example response:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.721,
  "risk_level": "High"
}
```

The model was trained on 19 encoded features. The API accepts a compact payload by applying sensible defaults for optional encoded fields.

Full feature order used by the model:

```text
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

## Test the API

With the FastAPI server running:

```bash
python src/predict.py
```

Expected output format:

```python
{'churn_prediction': 1, 'churn_probability': 0.721, 'risk_level': 'High'}
```

## Important Notes

- The dataset is not committed to GitHub because `data/` is ignored.
- The trained model is not committed because `models/` is ignored.
- MLflow local outputs are not committed because `mlruns/` and `mlflow.db` are ignored.
- To run the API after cloning, download the dataset and run `python src/train.py` first.
- Current categorical encoding uses `LabelEncoder`, so API inputs must use the same encoded integer values used during training.

## GitHub Push

Remote:

```text
git@github.com:bilaloit786/Churn-Prediction-with-Full-MLOps-Pipeline.git
```

Push command:

```bash
git push origin main
```

## Next Improvements

- Replace label encoding with a saved preprocessing pipeline.
- Save encoders alongside the model.
- Add Docker support.
- Add automated tests.
- Add GitHub Actions for linting and test runs.
- Register the best model in the MLflow Model Registry.
- Add a production-ready inference pipeline with input validation for raw categorical strings.
