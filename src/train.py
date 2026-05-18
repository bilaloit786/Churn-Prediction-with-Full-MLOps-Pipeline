"""Training entrypoint for the churn model."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


DATA_PATH = Path("data/telco.csv")
KAGGLE_DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.pkl")

EXPERIMENT_NAME = "churn-prediction"
RUN_NAME = "xgboost-baseline"


def load_data():
    dataset_path = DATA_PATH if DATA_PATH.exists() else KAGGLE_DATA_PATH
    return pd.read_csv(dataset_path)


def preprocess_data(df):
    # Drop customer ID - useless
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges - it's object, should be float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Log everything to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("resampling", "SMOTE")
        mlflow.log_param("smote_random_state", 42)
        mlflow.log_metric("train_rows_before_smote", len(X_train))
        mlflow.log_metric("train_rows_after_smote", len(X_train_bal))
        mlflow.log_metric("train_churn_before_smote", int(y_train.sum()))
        mlflow.log_metric("train_churn_after_smote", int(y_train_bal.sum()))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    return model


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model)


if __name__ == "__main__":
    main()
