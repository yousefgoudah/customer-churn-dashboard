from fastapi import FastAPI
import pandas as pd
import joblib

# create FastAPI app
app = FastAPI(title="Churn Prediction API")

# load saved model
model = joblib.load("models/churn_model.pkl")

# load saved training columns
columns = joblib.load("models/model_columns.pkl")


# homepage route
@app.get("/")
def home():
    return {"message": "Churn API Running Successfully"}


# prediction route
@app.post("/predict")
def predict(data: dict):

    # convert JSON input into dataframe
    df = pd.DataFrame([data])

    # feature engineering (same as training)
    df["AvgMonthlyValue"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    df["IsLongTerm"] = (df["Tenure Months"] > 24).astype(int)
    df["HighCharges"] = (df["Monthly Charges"] > 70).astype(int)

    # convert text columns to encoded columns
    df = pd.get_dummies(df)

    # match training columns exactly
    df = df.reindex(columns=columns, fill_value=0)

    # make prediction
    pred = int(model.predict(df)[0])

    # probability
    proba = float(model.predict_proba(df)[0][1])

    return {
        "prediction": pred,
        "churn_probability": round(proba, 4)
    }