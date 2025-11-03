import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# ✅ Load Saved Model Components
# ------------------------------
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
best_threshold = joblib.load("threshold.pkl")

# ------------------------------
# ✅ Helper Function for Prediction
# ------------------------------
def predict_diabetes(data_dict):
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])

    # ✅ Add engineered features
    df["Glucose_BMI_Ratio"] = df["Glucose"] / (df["BMI"] + 1)
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["AgeGroup"] = pd.cut(
        df["Age"], 
        bins=[20, 30, 40, 50, 60, 200],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # ✅ Impute values
    df_imp = imputer.transform(df)

    # ✅ Scale values
    df_scaled = scaler.transform(df_imp)

    # ✅ Predict probability
    prob = model.predict_proba(df_scaled)[0][1]

    # ✅ Apply threshold
    prediction = int(prob >= best_threshold)

    return prediction, round(prob, 4)

# ------------------------------
# ✅ Streamlit UI
# ------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Provide patient details to predict diabetes.")

# User Inputs
preg = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
bp = st.number_input("Blood Pressure", min_value=0, value=70)
skin = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):
    user_input = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    pred, prob = predict_diabetes(user_input)

    st.subheader("✅ Prediction Result")
    if pred == 1:
        st.error(f"🩸 **High Risk: Diabetes Detected** (probability: {prob})")
    else:
        st.success(f"✅ **Low Risk: No Diabetes** (probability: {prob})")
