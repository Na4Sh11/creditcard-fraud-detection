import joblib
import pandas as pd

# Load model
try:
    model = joblib.load('model/fraud_model.pkl')
except ModuleNotFoundError as e:
    print("Missing dependency:", e)
    
def predict_transaction(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability
