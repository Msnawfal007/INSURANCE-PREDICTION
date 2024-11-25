import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('insurance.csv')

# Preprocessing
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Prepare features (X) and target (y)
X = data.drop('charges', axis=1)
y = data['charges']
X_transformed = preprocessor.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Insurance Charges Prediction ❄️")
st.write("Use this tool to predict **medical insurance charges** based on user input. Fill in the form below:")

# User Input
age = st.number_input(
    "Age", 
    min_value=1, 
    max_value=100, 
    value=1,  # Default realistic value
    step=1
)

sex = st.selectbox(
    "Sex", 
    ["male", "female"], 
    index=0  # Default is 'male'
)

bmi = st.number_input(
    "BMI (Body Mass Index)", 
    min_value=10.0, 
    max_value=50.0, 
    value=10.0,  # Default realistic value
    step=0.1, 
    format="%.1f"
)

children = st.number_input(
    "Number of Children", 
    min_value=0, 
    max_value=10, 
    value=0,  # Default is 0
    step=1
)

smoker = st.selectbox(
    "Smoker", 
    ["yes", "no"], 
    index=1  # Default is 'no'
)

region = st.selectbox(
    "Region", 
    ["southwest", "southeast", "northwest", "northeast"], 
    index=0  # Default is 'southwest'
)

# Prepare user input for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

input_transformed = preprocessor.transform(input_data)

# Predict and display
if st.button("Predict"):
    prediction = model.predict(input_transformed)
    st.success(f"💰 Predicted Insurance Charges: **${prediction[0]:,.2f}**")
    st.snow()  # Trigger the snow effect

# Footer
st.write("Developed by Nawfal. Powered by **Streamlit** and **Machine Learning**.")
