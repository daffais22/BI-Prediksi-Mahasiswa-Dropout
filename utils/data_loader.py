import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

@st.cache_resource
def load_model():
    """Load trained model, scaler, and feature columns"""
    try:
        model = joblib.load('../models/best_dropout_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        with open('../models/feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_dataset():
    """Load clean dataset"""
    try:
        df = pd.read_excel('../data/clean_dataset.xlsx')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

@st.cache_data
def load_model_evaluation():
    """Load model evaluation results"""
    try:
        with open('../models/model_evaluation.json', 'r') as f:
            evaluation = json.load(f)
        return evaluation
    except Exception as e:
        st.warning("⚠️ Model evaluation not found. Please run the training notebook first.")
        return None