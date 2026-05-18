"""FastAPI application for churn prediction."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


MODEL_PATH = Path("models/churn_model.pkl")
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


app = FastAPI(title="Churn Prediction API")
model = joblib.load(MODEL_PATH)

FRONTEND_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Churn Prediction</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --surface: #ffffff;
      --line: #dbe3ee;
      --text: #172033;
      --muted: #617089;
      --primary: #0f766e;
      --primary-dark: #115e59;
      --danger: #b91c1c;
      --warning: #b45309;
      --success: #15803d;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 30px;
      line-height: 1.15;
    }

    p {
      margin: 0;
      color: var(--muted);
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.8fr);
      gap: 20px;
      align-items: start;
    }

    form,
    .result {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(23, 32, 51, 0.06);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    input,
    select {
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
    }

    input:focus,
    select:focus {
      outline: 3px solid rgba(15, 118, 110, 0.18);
      border-color: var(--primary);
    }

    button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      margin-top: 18px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--primary);
      color: #fff;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }

    button:hover {
      background: var(--primary-dark);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .result {
      position: sticky;
      top: 24px;
    }

    .result h2 {
      margin: 0 0 16px;
      font-size: 18px;
    }

    .metric {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 13px 0;
      border-top: 1px solid var(--line);
    }

    .metric span:first-child {
      color: var(--muted);
    }

    .metric span:last-child {
      font-weight: 800;
    }

    .risk-high {
      color: var(--danger);
    }

    .risk-medium {
      color: var(--warning);
    }

    .risk-low {
      color: var(--success);
    }

    .error {
      margin-top: 14px;
      color: var(--danger);
      font-weight: 650;
    }

    @media (max-width: 860px) {
      header,
      .layout {
        display: block;
      }

      .grid {
        grid-template-columns: 1fr;
      }

      .result {
        position: static;
        margin-top: 18px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Churn Prediction</h1>
        <p>Enter customer details and estimate churn risk.</p>
      </div>
      <p><a href="/docs">API docs</a></p>
    </header>

    <section class="layout">
      <form id="prediction-form">
        <div class="grid">
          <label>Tenure
            <input name="tenure" type="number" min="0" step="1" value="12" required>
          </label>
          <label>Monthly Charges
            <input name="MonthlyCharges" type="number" min="0" step="0.01" value="65.00" required>
          </label>
          <label>Total Charges
            <input name="TotalCharges" type="number" min="0" step="0.01" value="780.00" required>
          </label>
          <label>Contract
            <select name="Contract">
              <option value="0">Month-to-month</option>
              <option value="1">One year</option>
              <option value="2">Two year</option>
            </select>
          </label>
          <label>Payment Method
            <select name="PaymentMethod">
              <option value="0">Bank transfer</option>
              <option value="1">Credit card</option>
              <option value="2" selected>Electronic check</option>
              <option value="3">Mailed check</option>
            </select>
          </label>
          <label>Internet Service
            <select name="InternetService">
              <option value="0">DSL</option>
              <option value="1" selected>Fiber optic</option>
              <option value="2">No internet</option>
            </select>
          </label>
          <label>Senior Citizen
            <select name="SeniorCitizen">
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select>
          </label>
          <label>Partner
            <select name="Partner">
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select>
          </label>
          <label>Dependents
            <select name="Dependents">
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select>
          </label>
          <label>Phone Service
            <select name="PhoneService">
              <option value="0">No</option>
              <option value="1" selected>Yes</option>
            </select>
          </label>
          <label>Paperless Billing
            <select name="PaperlessBilling">
              <option value="0">No</option>
              <option value="1" selected>Yes</option>
            </select>
          </label>
          <label>Gender
            <select name="gender">
              <option value="0" selected>Female</option>
              <option value="1">Male</option>
            </select>
          </label>
        </div>
        <button type="submit">Predict churn</button>
        <div id="error" class="error" role="alert"></div>
      </form>

      <aside class="result" aria-live="polite">
        <h2>Prediction Result</h2>
        <div class="metric">
          <span>Prediction</span>
          <span id="prediction">Waiting</span>
        </div>
        <div class="metric">
          <span>Probability</span>
          <span id="probability">--</span>
        </div>
        <div class="metric">
          <span>Risk level</span>
          <span id="risk">--</span>
        </div>
      </aside>
    </section>
  </main>

  <script>
    const form = document.querySelector("#prediction-form");
    const errorBox = document.querySelector("#error");
    const prediction = document.querySelector("#prediction");
    const probability = document.querySelector("#probability");
    const risk = document.querySelector("#risk");

    const defaults = {
      MultipleLines: 0,
      OnlineSecurity: 0,
      OnlineBackup: 0,
      DeviceProtection: 0,
      TechSupport: 0,
      StreamingTV: 0,
      StreamingMovies: 0,
    };

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      errorBox.textContent = "";
      const submitButton = form.querySelector("button");
      submitButton.disabled = true;

      const formData = new FormData(form);
      const payload = { ...defaults };
      for (const [key, value] of formData.entries()) {
        payload[key] = Number(value);
      }

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error("Prediction request failed");
        }

        const data = await response.json();
        prediction.textContent = data.churn_prediction === 1 ? "Will churn" : "Will stay";
        probability.textContent = `${Math.round(data.churn_probability * 1000) / 10}%`;
        risk.textContent = data.risk_level;
        risk.className = `risk-${data.risk_level.toLowerCase()}`;
      } catch (error) {
        errorBox.textContent = "Could not generate a prediction. Check that the API is running.";
      } finally {
        submitButton.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class CustomerData(BaseModel):
    gender: int = 0
    SeniorCitizen: int = 0
    Partner: int = 0
    Dependents: int = 0
    tenure: float
    PhoneService: int = 1
    MultipleLines: int = 0
    InternetService: int
    OnlineSecurity: int = 0
    OnlineBackup: int = 0
    DeviceProtection: int = 0
    TechSupport: int = 0
    StreamingTV: int = 0
    StreamingMovies: int = 0
    Contract: int
    PaperlessBilling: int = 1
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


def get_risk_level(probability):
    if probability > 0.7:
        return "High"
    if probability > 0.4:
        return "Medium"
    return "Low"


@app.get("/", response_class=HTMLResponse)
def root():
    return FRONTEND_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CustomerData):
    features = pd.DataFrame([[getattr(data, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": get_risk_level(probability),
    }
