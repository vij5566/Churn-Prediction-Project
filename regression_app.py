import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Customer Salary Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model('models/model.keras')

    with open('utils/geo_encoder.pkl', 'rb') as f:
        one_hot_en = pickle.load(f)

    with open('utils/scaler_reg.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('utils/label_encoder.pkl','rb') as f:
        label_en_gender = pickle.load(f)

    return model, one_hot_en, scaler, label_en_gender


model, one_hot_en, scaler, label_en_gender = load_all()

# ------------------ HEADER ------------------ #
st.markdown("""
    <h1 style='text-align: center; color:#2E86C1;'>
        💰 Customer Salary Prediction Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Predict customer's estimated salary using AI
    </p>
""", unsafe_allow_html=True)

st.divider()

# ------------------ INPUT FORM ------------------ #
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Info")
        geography = st.selectbox('Geography', one_hot_en.categories_[0])
        gender = st.selectbox('Gender', label_en_gender.classes_)
        age = st.slider('Age', 18, 92)

    with col2:
        st.subheader("💳 Account Info")
        credit_score = st.number_input('Credit Score', value=600)
        balance = st.number_input(
            '💰 Balance (₹)',
            min_value=0.0,
            max_value=1000000.0,
            value=50000.0,
            step=100.0
        )
        tenure = st.slider('Tenure', 0, 10)

    with col3:
        st.subheader("📊 Activity Info")
        num_of_products = st.slider('Number of Products', 1, 4)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Is Active Member', [0, 1])

    submit = st.form_submit_button("🚀 Predict Salary")

# ------------------ PREDICTION ------------------ #
if submit:

    # Input DataFrame (NO EstimatedSalary here ❗)
    input_df = pd.DataFrame({
        'CreditScore':[credit_score],
        'Gender':[gender],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'Geography':[geography]
    })

    # ------------------ ENCODING ------------------ #
    geo_encoded = one_hot_en.transform(input_df[['Geography']])
    geo_df = pd.DataFrame(
        geo_encoded.toarray(),
        columns=one_hot_en.get_feature_names_out(['Geography'])
    )

    input_df = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)

    input_df['Gender'] = label_en_gender.transform(input_df['Gender'])

    # ------------------ FEATURE ALIGNMENT ------------------ #
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # ------------------ SCALING ------------------ #
    input_scaled = scaler.transform(input_df)

    # ------------------ PREDICTION ------------------ #
    with st.spinner("🔍 Predicting salary..."):
        prediction = model.predict(input_scaled)
        salary = prediction[0][0]

    st.divider()

    col1, col2 = st.columns(2)

    # ------------------ INPUT DISPLAY ------------------ #
    with col1:
        st.subheader("📌 Customer Data")
        st.dataframe(input_df, use_container_width=True)

    # ------------------ RESULT ------------------ #
    with col2:
        st.subheader("📈 Prediction Result")

        st.metric("Predicted Salary", f"₹ {salary:,.2f}")

        if salary > 100000:
            st.success("🌟 High earning customer")
        elif salary > 50000:
            st.warning("⚖️ متوسط earning customer")
        else:
            st.error("⚠️ Low earning customer")

# ------------------ FOOTER ------------------ #
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)