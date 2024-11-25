import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('insurance.csv')

# Preprocess the data
@st.cache
def preprocess_data(data, categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

# Train the model and save it
def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model and encoder
    joblib.dump(model, "insurance_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    st.write("Model and preprocessor saved successfully!")
    return model

# Load the model and encoder
def load_model_and_encoder():
    model = joblib.load("insurance_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

# Main Streamlit App
st.title("Insurance Cost Prediction App")

# Load data
data = load_data()
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

# Train or Load Model
if st.sidebar.button("Train Model"):
    X_transformed, y, preprocessor = preprocess_data(data, categorical_features, numeric_features)
    model = train_and_save_model(X_transformed, y)
else:
    model, preprocessor = load_model_and_encoder()

st.subheader("Make a Prediction")
# User Inputs
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10, 50, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
sex = st.selectbox("Sex", options=["male", "female"])
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Prepare user input for prediction
user_input = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex': [sex],
    'smoker': [smoker],
    'region': [region]
})

user_transformed = preprocessor.transform(user_input)
prediction = model.predict(user_transformed)

# Display the result
if st.button("Predict"):
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
