import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# Cover / Header
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1f2937, #111827);
        padding: 24px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 18px;
        color: white;">
        <h1 style="margin:0;">AI-Driven Network Traffic Anomaly Detection</h1>
        <p style="font-size:16px; color:#9ca3af; margin-top:6px;">
            Machine Learning + Streamlit Dashboard — UNSW-NB15 / CICIDS2017
        </p>
    </div>
""", unsafe_allow_html=True)



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

    # Footer
    st.markdown("""
        <hr style="margin-top:36px; border: 1px solid #374151;">
        <div style="text-align:center; color:#9ca3af;">
            <p>Built by <b>Kaan Sulkalar</b> — <a style="color:#60a5fa;" href="https://github.com/KaanSulkular/NetworkAnomalyDetection">GitHub</a></p>
            <p style="font-size:12px;">© 2025 · AI-Driven Network Anomaly Detection</p>
        </div>
    """, unsafe_allow_html=True)
