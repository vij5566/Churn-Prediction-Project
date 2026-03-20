import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Load model safely
#model = tf.keras.models.load_model('model.h5', compile=False)
model = tf.keras.models.load_model('models/model.keras')

with open('utils/geo_encoder.pkl', 'rb') as f:
    one_hot_en = pickle.load(f)

with open('utils/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('utils/label_encoder.pkl','rb') as f:
    label_en_gender = pickle.load(f)

# Title
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox('Geography', one_hot_en.categories_[0])
gender = st.sidebar.selectbox('Gender', label_en_gender.classes_)

age = st.sidebar.slider('Age',18,92)
balance = st.sidebar.number_input('Balance', min_value=0.0,max_value=1000000.0,value=0.0,step=100.0)
credit_score = st.sidebar.number_input('Credit Score', value=600)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000.0)
tenure = st.sidebar.slider('Tenure',0,10)
num_of_products = st.sidebar.slider('Number of Products',1,4)
has_cr_card = st.sidebar.selectbox('Has Credit Card',[0,1])
is_active_member = st.sidebar.selectbox('Is Active Member',[0,1])

# Input DataFrame
input_df = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    'Geography':[geography]
})

# Encoding
geo_encoded = one_hot_en.transform(input_df[['Geography']])
geo_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=one_hot_en.get_feature_names_out(['Geography'])
)

input_df = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)

input_df['Gender'] = label_en_gender.transform(input_df['Gender'])

# Scaling
input_scaled = scaler.transform(input_df)

# Layout for results
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Input Summary")
    st.write(input_df)

with col2:
    st.subheader("🔍 Prediction Result")

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        prediction_prob = prediction[0][0]

        # Progress bar (visual appeal)
        st.progress(int(prediction_prob * 100))

        st.metric("Churn Probability", f"{prediction_prob:.2f}")

        if prediction_prob > 0.5:
            st.error("⚠️ Customer is likely to leave")
        else:
            st.success("✅ Customer is likely to stay")

#st.markdown("---")
#st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)