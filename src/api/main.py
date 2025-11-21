from fastapi import FastAPI
import joblib
import pandas as pd
from src.utils.risk_logic import risk_category
from src.utils.recovery_recommendations import recovery_action

app = FastAPI()

# Load model
model = joblib.load("src/models/xgboost_model.pkl")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "CreditPathAI Risk API is running!"}

@app.post("/predict/")
def predict(data: dict):
    # Convert incoming JSON to DataFrame row
    df = pd.DataFrame([data])

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Risk category
    category = risk_category(prob)

    # Recommended action
    action = recovery_action(category)

    return {
        "probability_of_default": float(prob),
        "risk_category": category,
        "recommended_action": action
    }
