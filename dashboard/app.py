import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("AI-Driven Network Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model = joblib.load('../src/model_rf.pkl')
    preds = model.predict(df)
    df['Prediction'] = preds
    st.subheader("Prediction Results")
    st.write(df.head())
    fig = px.histogram(df, x='Prediction', color='Prediction', title="Anomaly Distribution")
    st.plotly_chart(fig)