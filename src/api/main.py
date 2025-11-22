from fastapi import FastAPI
import joblib
import pandas as pd
from src.utils.risk_logic import risk_category
from src.utils.recovery_recommendations import recovery_action
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

origins = [
    "https://legendary-space-sniffle-r7g5pvq6j552p56q-3000.app.github.dev",
    "http://localhost:3000",
]

class LoanData(BaseModel):
    income: float
    loan_amount: float
    credit_score: int
    ltv: float
    dtir1: float
    # ... all features used during model training

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    "risk_category": "Medium Risk",
    "probability": 0.45,
    "recommendation": "Send reminder SMS"
}

