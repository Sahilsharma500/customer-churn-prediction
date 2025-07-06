import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and preprocessing tools
model = load_model("model-training/model.h5")
scaler = pickle.load(open("model-training/scaler.pkl", "rb"))
le_gender = pickle.load(open("model-training/label_encoder_gender.pkl", "rb"))
ohe_geo = pickle.load(open("model-training/onehot_encoder_geo.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("🔍 Customer Churn Prediction")
st.markdown("Enter customer details below to predict whether the customer is likely to churn:")

# Input Form
with st.form("churn_form"):
    CreditScore = st.text_input("Credit Score")
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.text_input("Age")
    Tenure = st.text_input("Tenure")
    Balance = st.text_input("Balance")
    NumOfProducts = st.text_input("Number of Products")
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.text_input("Estimated Salary")

    submitted = st.form_submit_button("Predict")

# Prediction Logic
if submitted:
    try:
        # Convert string inputs to appropriate types
        CreditScore = float(CreditScore)
        Age = float(Age)
        Tenure = float(Tenure)
        Balance = float(Balance)
        NumOfProducts = int(NumOfProducts)
        EstimatedSalary = float(EstimatedSalary)

        # Check for valid ranges
        if any(val < 0 for val in [CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary]):
            st.error("All numeric inputs must be non-negative.")
        else:
            # Create DataFrame
            df = pd.DataFrame([{
                'CreditScore': CreditScore,
                'Geography': Geography,
                'Gender': Gender,
                'Age': Age,
                'Tenure': Tenure,
                'Balance': Balance,
                'NumOfProducts': NumOfProducts,
                'HasCrCard': HasCrCard,
                'IsActiveMember': IsActiveMember,
                'EstimatedSalary': EstimatedSalary
            }])

            # Encode Gender
            df['Gender'] = le_gender.transform(df['Gender'])

            # Encode Geography
            geo_encoded = ohe_geo.transform(df[['Geography']]).toarray()
            geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
            df = df.drop('Geography', axis=1)
            df = pd.concat([df, geo_df], axis=1)

            # Scale
            X_scaled = scaler.transform(df)

            # Predict
            pred = model.predict(X_scaled)[0][0]
            churn = pred >= 0.5

            st.subheader("Prediction Result")
            if churn:
                st.error("The customer is **likely to churn**.")
            else:
                st.success("The customer is **likely to stay**.")
            st.markdown(f"**Churn Probability:** `{pred:.2f}`")

    except ValueError:
        st.error("Please enter valid numeric values in all fields.")
    except Exception as e:
        st.error(f"Something went wrong:\n\n{e}")
