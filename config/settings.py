import streamlit as st
from pathlib import Path

def apply_page_config():
    """Apply page configuration"""
    st.set_page_config(
        page_title="Dashboard Prediksi Dropout Mahasiswa",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    css_dir = Path(__file__).parent / "styles"

    css_files = [
        "base.css",
        "components.css",
        "risk-indicators.css"
    ]

    combined_css = ""
    for css_file in css_files:
        css_path = css_dir / css_file
        if css_path.exists():
            with open(css_path) as f:
                combined_css += f.read() + "\n"

    st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)

# Constants
MENU_OPTIONS = [
    "🏠 Home",
    "📊 Dashboard Analitik",
    "🔮 Prediksi Individu",
    "📈 Analisis Mahasiswa",
    "ℹ️ Info Model"
]

RISK_COLORS = {
    'TINGGI': '#c62828',
    'SEDANG': '#ef6c00',
    'RENDAH': '#2e7d32'
}