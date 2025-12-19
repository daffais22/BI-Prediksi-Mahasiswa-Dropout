import pandas as pd
import numpy as np

def categorize_ipk(ipk):
    """Categorize IPK into performance levels (REVISED)"""
    if pd.isna(ipk):
        return 0
    if ipk >= 3.5:
        return 4  # Cum Laude
    elif ipk >= 3.0:
        return 3  # Sangat Memuaskan
    elif ipk >= 2.75:
        return 2  # Memuaskan
    elif ipk >= 2.0:
        return 1  # Cukup
    else:
        return 0  # Kurang (Berisiko)

def process_data(df):
    """Process raw data for prediction (REVISED)"""
    df_processed = df.copy()
    
    # Handle missing values first
    if 'IPK' in df_processed.columns:
        df_processed['IPK'].fillna(df_processed['IPK'].median(), inplace=True)
    if 'SKS' in df_processed.columns:
        df_processed['SKS'].fillna(df_processed['SKS'].median(), inplace=True)
    if 'Kehadiran' in df_processed.columns:
        df_processed['Kehadiran'].fillna(df_processed['Kehadiran'].median(), inplace=True)
    
    # Create target variable (REVISED - based on IPK AND Kehadiran)
    KEHADIRAN_THRESHOLD = 0.7
    IPK_THRESHOLD = 2.0
    
    df_processed['Target'] = df_processed.apply(
        lambda row: 1 if (row['Kehadiran'] < KEHADIRAN_THRESHOLD and row['IPK'] < IPK_THRESHOLD) else 0,
        axis=1
    )
    
    # Status encoding
    status_mapping = {
        'AKTIF': 0,
        'LULUS': 1,
        'CUTI': 2,
        'KELUAR': 3,
        'NON AKTIF': 4,
        'REGISTRASI': 5
    }
    df_processed['Status_Encoded'] = df_processed['Status'].map(status_mapping)
    
    # Status Risk (CUTI, KELUAR, NON AKTIF = High Risk)
    df_processed['Status_Risk'] = df_processed['Status'].apply(
        lambda x: 1 if x.upper() in ['CUTI', 'KELUAR', 'NON AKTIF'] else 0
    )
    
    # IPK Category
    df_processed['IPK_Category'] = df_processed['IPK'].apply(categorize_ipk)
    
    # IPK Risk (IPK < 2.0)
    df_processed['IPK_Risk'] = (df_processed['IPK'] < IPK_THRESHOLD).astype(int)
    
    # Kehadiran Category
    df_processed['Kehadiran_Category'] = pd.cut(
        df_processed['Kehadiran'], 
        bins=[0, 0.5, 0.7, 0.85, 1.0],
        labels=[0, 1, 2, 3],  # 0=Sangat Rendah, 1=Rendah, 2=Sedang, 3=Baik
        include_lowest=True
    )
    df_processed['Kehadiran_Category'] = pd.to_numeric(df_processed['Kehadiran_Category'], errors='coerce').fillna(0).astype(int)
    
    # Kehadiran Risk (Kehadiran < 70%)
    df_processed['Kehadiran_Risk'] = (df_processed['Kehadiran'] < KEHADIRAN_THRESHOLD).astype(int)
    
    # Combined Risk Score (0-4)
    df_processed['Risk_Score'] = (
        df_processed['IPK_Risk'] * 2 +  # IPK rendah = 2 poin
        df_processed['Kehadiran_Risk'] * 2 +  # Kehadiran rendah = 2 poin
        df_processed['Status_Risk'] * 1  # Status berisiko = 1 poin
    ).clip(upper=4)
    
    return df_processed