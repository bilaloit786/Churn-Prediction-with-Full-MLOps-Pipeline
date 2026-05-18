# Churn MLOps

Customer churn prediction project using FastAPI, XGBoost, SMOTE, and MLflow.

## Project Structure

```text
data/          # Local datasets, ignored by git
models/        # Trained model artifacts, ignored by git
notebooks/     # Experiment notebooks
src/
  api.py       # FastAPI app and frontend
  predict.py   # API test client
  train.py     # Training, MLflow tracking, and model saving
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Run API

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Test Prediction

```bash
python src/predict.py
```
