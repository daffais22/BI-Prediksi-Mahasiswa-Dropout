import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configurations
from config.settings import apply_page_config, apply_custom_css, MENU_OPTIONS

# Import utilities
from utils.data_loader import load_model, load_dataset, load_model_evaluation
from utils.preprocessor import process_data

# Import pages
from pages import home, analytics, prediction, analysis, model_info

def main():
    """Main application"""

    # Apply configurations
    apply_page_config()
    apply_custom_css()
    
    # Load data and model
    model, scaler, feature_cols = load_model()
    df = load_dataset()
    model_eval = load_model_evaluation()
    
    # Check if data and model loaded successfully
    if df is None or model is None:
        st.error("❌ Failed to load data or model. Please check the file paths.")
        st.info("""
        **Troubleshooting:**
        1. Make sure `clean_dataset.xlsx` exists in `../data/` folder
        2. Make sure model files exist in `../models/` folder
        3. Run the training notebook (`test.ipynb`) to generate model files
        """)
        return
    
    # Process data
    df_processed = process_data(df)
    
    # Sidebar navigation
    st.sidebar.title("🎓 Navigation")
    
    
    menu = st.sidebar.radio(
        "Pilih Menu:",
        MENU_OPTIONS,
        label_visibility="collapsed"
    )
    
    
    # Show info in sidebar
    st.sidebar.info(f"""
    **📊 Dataset Info**
    - Total Mahasiswa: {len(df)}
    - Dropout Rate: {(df_processed['Target'].sum()/len(df)*100):.1f}%
    - Avg IPK: {df['IPK'].mean():.2f}
    """)
    
    # Route to appropriate page
    if menu == "🏠 Home":
        home.show(df, df_processed)
    
    elif menu == "📊 Dashboard Analitik":
        analytics.show(df, model, scaler)
    
    elif menu == "🔮 Prediksi Individu":
        prediction.show(model, scaler)
    
    elif menu == "📈 Analisis Mahasiswa":
        analysis.show(df, model, scaler)
    
    elif menu == "ℹ️ Info Model":
        model_info.show(model_eval)

if __name__ == "__main__":
    main()