
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="AI-Driven Network Anomaly Detection", layout="wide")
st.title("AI-Driven Network Traffic Anomaly Detection 🚦")

with st.sidebar:
    st.header("🔧 Ayarlar & Yardım")
    st.markdown(
        "- **Adımlar:** EDA → Model → Tahmin\n"
        "- **Etiketli veri:** `label` veya seçtiğiniz sütun\n"
        "- **Kaydet/Yükle:** Model & scaler dosyaları"
    )
    st.divider()
    st.caption("Mac/Win/Linux üzerinde çalışır. Python 3.11+ önerilir.")

@st.cache_data(show_spinner=False)
def read_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

def split_features(df: pd.DataFrame, label_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col and label_col in df.columns:
        X = df.drop(columns=[label_col])
        y = df[label_col]
        return X, y
    else:
        return df.copy(), None

def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("lbl", "passthrough")  # Label encoding will be applied ad-hoc
    ])
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", "passthrough", cat_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols

def encode_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    df_enc = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        encoders[c] = le
    return df_enc, encoders

def plot_confusion(cm, labels=("Normal","Attack")):
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
        text=cm, texttemplate="%{text}", hoverinfo="skip"
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_pr(y_true, y_score):
    try:
        roc = roc_auc_score(y_true, y_score)
        fpr = np.linspace(0, 1, 200)
        # Build a pseudo ROC curve by threshold sweep using percentiles
        thresholds = np.percentile(y_score, np.linspace(0, 100, 200))
        tprs, fprs = [], []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp+fn)>0 else 0.0
            fpr_ = fp / (fp + tn) if (fp+tn)>0 else 0.0
            tprs.append(tpr); fprs.append(fpr_)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=fprs, y=tprs, mode="lines", name=f"ROC (AUC~{roc:.3f})"))
        fig1.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
        fig1.update_layout(title="ROC Curve (approx.)", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig1, use_container_width=True)

        pr, rc, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rc, pr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
        fig2.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.info(f"ROC/PR çizimi atlandı: {e}")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Model", "🔎 Tahmin"])

with tab1:
    st.subheader("Keşifsel Veri Analizi (EDA)")
    up = st.file_uploader("Network CSV yükleyin", type=["csv"], key="eda")
    if up:
        df = read_csv(up)
        st.write("Örnek:", df.head())
        st.write("Şekil:", df.shape)
        st.write("Sütun tipleri:", df.dtypes)

        # Hedef sütun seçimi
        cols = df.columns.tolist()
        label_guess = "label" if "label" in cols else None
        label_col = st.selectbox("Etiket (opsiyonel)", [None] + cols, index=(cols.index(label_guess)+1) if label_guess in cols else 0)
        X, y = split_features(df, label_col)

        # Sınıf dağılımı
        if y is not None:
            st.markdown("**Sınıf Dağılımı**")
            st.bar_chart(y.value_counts())

        # Sayısal sütun dağılımları
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        if num_cols:
            chosen = st.selectbox("Histogram sütunu", num_cols, index=0)
            fig = px.histogram(X, x=chosen, nbins=50, title=f"Dağılım: {chosen}")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Eğitimi ve Değerlendirme")
    up = st.file_uploader("Eğitim için CSV yükleyin", type=["csv"], key="train")
    model_type = st.selectbox("Model", ["RandomForest (supervised)", "IsolationForest (unsupervised)"])
    test_size = st.slider("Test oranı", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("random_state", 0, 9999, 42, step=1)
    save_prefix = st.text_input("Kayıt adı (model/scaler dosya prefix)", "nad_model")

    col1, col2 = st.columns(2)
    with col1:
        train_btn = st.button("⚡ Eğitimi Başlat")
    with col2:
        load_btn = st.button("📥 Kayıtlı Model/Scaler Yükle")

    if load_btn:
        try:
            clf = joblib.load(f"../src/{save_prefix}_model.joblib")
            scaler = joblib.load(f"../src/{save_prefix}_scaler.joblib")
            enc_map = joblib.load(f"../src/{save_prefix}_encoders.joblib")
            st.success("Model ve scaler yüklendi.")
            st.session_state["clf"] = clf
            st.session_state["scaler"] = scaler
            st.session_state["enc_map"] = enc_map
        except Exception as e:
            st.error(f"Yükleme hatası: {e}")

    if up and train_btn:
        df = read_csv(up)
        cols = df.columns.tolist()
        label_guess = "label" if "label" in cols else None
        label_col = st.selectbox("Etiket kolonu", cols, index=cols.index(label_guess) if label_guess in cols else 0, key="label_sel")
        X, y = split_features(df, label_col)

        pre, num_cols, cat_cols = build_preprocess(X)
        X_enc, enc_map = encode_categoricals(X, cat_cols)
        # Scale only numerical columns; keep encoded cats as is
        scaler = StandardScaler()
        X_num = X_enc[num_cols].values if num_cols else np.empty((len(X_enc), 0))
        X_num_scaled = scaler.fit_transform(X_num) if X_num.size else X_num
        X_final = np.hstack([X_num_scaled, X_enc[[c for c in cat_cols]].values]) if cat_cols else X_num_scaled

        if model_type.startswith("RandomForest"):
            X_tr, X_te, y_tr, y_te = train_test_split(X_final, y, test_size=test_size, random_state=random_state, stratify=y)
            clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            try:
                y_prob = clf.predict_proba(X_te)[:,1]
            except Exception:
                y_prob = y_pred

            st.code(classification_report(y_te, y_pred))
            cm = confusion_matrix(y_te, y_pred)
            plot_confusion(cm, labels=("0","1"))
            try:
                roc = roc_auc_score(y_te, y_prob)
                st.metric("ROC-AUC", f"{roc:.4f}")
            except Exception:
                st.info("ROC-AUC hesaplanamadı.")
            plot_roc_pr(y_te, y_prob)

            # Feature importance (approx.)
            try:
                importances = clf.feature_importances_
                imp_fig = px.bar(x=np.arange(len(importances)), y=importances, labels={"x":"Feature Index","y":"Importance"}, title="Feature Importances")
                st.plotly_chart(imp_fig, use_container_width=True)
            except Exception:
                pass

            # Save artifacts
            os.makedirs("../src", exist_ok=True)
            joblib.dump(clf, f"../src/{save_prefix}_model.joblib")
            joblib.dump(scaler, f"../src/{save_prefix}_scaler.joblib")
            joblib.dump(enc_map, f"../src/{save_prefix}_encoders.joblib")
            st.success(f"Kaydedildi: ../src/{save_prefix}_model.joblib (ve scaler, encoders)")

            st.session_state["clf"] = clf
            st.session_state["scaler"] = scaler
            st.session_state["enc_map"] = enc_map
            st.session_state["num_cols"] = num_cols
            st.session_state["cat_cols"] = cat_cols

        else:
            # IsolationForest (unsupervised) – ignore label for training
            X_all = X_final
            iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1)
            iso.fit(X_all)
            scores = -iso.score_samples(X_all)
            st.write("Skor istatistikleri:", {"mean":float(np.mean(scores)), "std":float(np.std(scores))})
            fig = px.histogram(x=scores, nbins=50, title="IsolationForest Anomaly Scores")
            st.plotly_chart(fig, use_container_width=True)

            os.makedirs("../src", exist_ok=True)
            joblib.dump(iso, f"../src/{save_prefix}_model.joblib")
            joblib.dump(scaler, f"../src/{save_prefix}_scaler.joblib")
            joblib.dump(enc_map, f"../src/{save_prefix}_encoders.joblib")
            st.success(f"Kaydedildi: ../src/{save_prefix}_model.joblib (ve scaler, encoders)")

            st.session_state["clf"] = iso
            st.session_state["scaler"] = scaler
            st.session_state["enc_map"] = enc_map
            st.session_state["num_cols"] = num_cols
            st.session_state["cat_cols"] = cat_cols

with tab3:
    st.subheader("Tahmin / Skorlama")
    up = st.file_uploader("Tahmin için CSV yükleyin", type=["csv"], key="pred")
    model_loaded = ("clf" in st.session_state) and ("scaler" in st.session_state) and ("enc_map" in st.session_state)
    if up and model_loaded:
        df = read_csv(up)
        enc_map = st.session_state["enc_map"]
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(set(df.columns) - set(num_cols))

        # Apply encoders (unseen labels -> new codes)
        df_enc = df.copy()
        for c, le in enc_map.items():
            if c in df_enc.columns:
                vals = df_enc[c].astype(str).values
                known = list(le.classes_)
                new_classes = np.setdiff1d(np.unique(vals), known)
                if len(new_classes) > 0:
                    le.classes_ = np.concatenate([le.classes_, new_classes])
                df_enc[c] = le.transform(vals)

        # Scale numerics
        scaler = st.session_state["scaler"]
        X_num = df_enc.select_dtypes(include=[np.number]).values
        X_num_scaled = scaler.transform(X_num) if X_num.size else X_num
        X_final = X_num_scaled

        clf = st.session_state["clf"]
        try:
            preds = clf.predict(X_final)
        except Exception:
            # Some models require thresholding
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_final)
            else:
                scores = getattr(clf, "score_samples", lambda z: np.zeros(len(z)))(X_final)
            thresh = np.percentile(scores, 95)
            preds = (scores >= thresh).astype(int)

        out = df.copy()
        out["Prediction"] = preds
        st.dataframe(out.head(50), use_container_width=True)

        fig = px.histogram(out, x="Prediction", title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Sonuçları CSV olarak indir",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Model ve scaler yüklü olmalı. 'Model' sekmesinde eğittikten sonra buraya dönün veya kayıtlı dosyaları yükleyin.")
