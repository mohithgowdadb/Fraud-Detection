from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

_model = joblib.load(BASE_DIR / "fraud_model.joblib")
_preprocessor = joblib.load(BASE_DIR / "fraud_preprocessor.joblib")
THRESHOLD = float((BASE_DIR / "fraud_threshold.txt").read_text())


def add_time_features(df):
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Transaction_Hour"] = df["Timestamp"].dt.hour
    df["Day_Of_Week"] = df["Timestamp"].dt.dayofweek
    df["Is_Weekend_Corrected"] = (df["Day_Of_Week"] >= 5).astype(int)
    return df


def predict_transaction(transaction: dict) -> dict:
    
    df = pd.DataFrame([transaction])
    df = add_time_features(df)

   
    X = _preprocessor.transform(df)
    probability = float(_model.predict_proba(X)[0, 1])

    return {
        "fraud_probability": round(probability, 4),
        "is_flagged": probability >= THRESHOLD,
        "threshold": round(THRESHOLD, 4),
    }