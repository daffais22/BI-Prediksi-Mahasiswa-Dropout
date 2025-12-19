import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ambil root project
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model():
    try:
        model = joblib.load(BASE_DIR / "models" / "best_dropout_model.pkl")
        scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
        with open(BASE_DIR / "models" / "feature_columns.json") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_excel(BASE_DIR / "data" / "clean_dataset.xlsx")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

@st.cache_data
def load_model_evaluation():
    try:
        with open(BASE_DIR / "models" / "model_evaluation.json") as f:
            return json.load(f)
    except:
        st.warning("⚠️ Model evaluation not found.")
        return None
