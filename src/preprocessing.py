import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    label_cols = df.select_dtypes(include=['object']).columns
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled