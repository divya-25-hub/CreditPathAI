import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.utils.risk_logic import risk_category
from src.utils.recommendation_engine import recommend_action
from src.utils.recovery_recommendations import recovery_action
import pickle

CLEAN_DATA = "data/processed/cleaned_Loan_Default.csv"
MODEL_SAVE_XGB = "src/models/xgboost_model.pkl"
MODEL_SAVE_LGB = "src/models/lightgbm_model.pkl"

def load_data():
    print("📌 Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_DATA)
    return df

def preprocess(df):
    print("📌 Preprocessing...")

    X = df.drop("status", axis=1)
    y = df["status"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, X_test, y_train, y_test):
    print("🚀 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("📌 XGBoost Report:")
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))

    with open(MODEL_SAVE_XGB, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ XGBoost model saved to {MODEL_SAVE_XGB}")

def train_lightgbm(X_train, X_test, y_train, y_test):
    print("🚀 Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    prob = model.predict_proba(X_test)[0][1]
    category = risk_category(prob)
    recommendation = recommend_action(category)
    recommendation = recovery_action(category)
    print("Example Risk Category:", category)
    print("Recommended Action:", recommendation)
    print("Recovery Recommendation:", recommendation)

    print("📌 LightGBM Report:")
    print(classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))

    with open(MODEL_SAVE_LGB, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ LightGBM model saved to {MODEL_SAVE_LGB}")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    train_xgboost(X_train, X_test, y_train, y_test)
    train_lightgbm(X_train, X_test, y_train, y_test)

