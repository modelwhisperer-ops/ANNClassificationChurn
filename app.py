import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import tensorflow as tf, streamlit as st

# ANN Implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
from tensorflow.keras.models import load_model

# Load the trained model, scaler, OHE pickle files
model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Deploying the model using Streamlit
# After loading the model, scaler and encoders

st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
tenure = st.slider('Tenure', 0, 10, 5)
balance = st.number_input('Balance', min_value=0.0, value=50000.0)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
credit_score = st.number_input('Credit Score', min_value=0, value=600)

# Prepare the input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# One Hot Encode Geography
geo_encoder_st = onehot_encoder_geo.transform([[geography]]) 
geo_encoded_st_df = pd.DataFrame(geo_encoder_st.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df, geo_encoded_st_df], axis=1)
# input_df = input_df.drop(['Geography'], axis=1)

# Prediction
model_input_st = scaler.transform(input_df)
prediction_st = model.predict(model_input_st)
print(prediction_st)
print("Prediction Probability:", st.write(prediction_st[0][0]))
print("Prediction:", st.write("Exited") if prediction_st[0][0] > 0.5 else st.write("Not Exited"))




