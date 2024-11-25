import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
data = pd.read_csv('insurance.csv')

# Preprocess the data
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']
target = 'charges'

# Define ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Leave numeric features as is
)

# Prepare features (X) and target (y)
X = data.drop(target, axis=1)
y = data[target]

# Preprocess features
X_transformed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.set_page_config(page_title="Insurance Charges Prediction", layout="wide")
st.title("Insurance Charges Prediction")

# Add a background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.w3schools.com/w3images/mountains.jpg");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True
)

st.write("Enter the details below to predict insurance charges:")

# Input features with number inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])

# Transform input for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocess input
input_transformed = preprocessor.transform(input_data)

# Add a loading spinner while predicting
with st.spinner('Predicting charges...'):
    if st.button("Predict"):
        prediction = model.predict(input_transformed)[0]
        
        # Apply money effect
        st.subheader("Predicted Insurance Charges")
        st.markdown(
            f"""
            <p style="font-size: 30px; color: #FF5733; font-weight: bold; transition: all 0.5s ease-in-out; text-align: center;">
                ${prediction:,.2f}
            </p>
            """, unsafe_allow_html=True
        )
