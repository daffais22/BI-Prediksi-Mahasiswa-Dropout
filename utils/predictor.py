import numpy as np
import pandas as pd

def predict_dropout_risk(model, scaler, ipk, kehadiran, status):
    """
    Predict dropout risk for a student (REVISED - 3 features only)
    
    Parameters:
    -----------
    model : trained model
    scaler : fitted scaler
    ipk : float (0-4)
    kehadiran : float (0-1)
    status : str ('AKTIF', 'LULUS', 'CUTI', 'KELUAR', 'NON AKTIF', 'REGISTRASI')
    
    Returns:
    --------
    dict : prediction results
    """
    # Normalize kehadiran if in percentage
    if kehadiran > 1:
        kehadiran = kehadiran / 100
    
    # Calculate Status Risk (1 = high risk, 0 = low risk)
    status_risk = 1 if status.upper() in ['CUTI', 'KELUAR', 'NON AKTIF'] else 0
    
    # Create feature array (MUST MATCH: IPK, Kehadiran, Status_Risk)
    features = np.array([[ipk, kehadiran, status_risk]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    model_prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Determine actual dropout condition based on business rules
    actual_dropout = (kehadiran < 0.7 and ipk < 2.0)
    
    # ENHANCED LOGIC: Use business rule + model probability
    if actual_dropout and probability[1] > 0.05:
        final_prediction = 1
    elif probability[1] > 0.75:
        final_prediction = 1
    else:
        final_prediction = 0
    
    # Adjust risk level based on actual condition
    if actual_dropout:
        if probability[1] > 0.3 or status_risk == 1:
            risk_level = 'TINGGI'
        elif probability[1] > 0.05:
            risk_level = 'SEDANG'
        else:
            risk_level = 'SEDANG'
    else:
        if probability[1] > 0.7:
            risk_level = 'TINGGI'
        elif probability[1] > 0.4:
            risk_level = 'SEDANG'
        else:
            risk_level = 'RENDAH'
    
    # Prepare result
    result = {
        'prediction': 'RISIKO DROPOUT' if final_prediction == 1 else 'TIDAK BERISIKO',
        'actual_dropout_condition': actual_dropout,
        'dropout_probability': float(probability[1]),
        'safe_probability': float(probability[0]),
        'risk_level': risk_level,
        'details': {
            'ipk': ipk,
            'ipk_status': '❌ RENDAH (< 2.0)' if ipk < 2.0 else '✅ BAIK',
            'kehadiran': f"{kehadiran*100:.1f}%",
            'kehadiran_status': '❌ RENDAH (< 70%)' if kehadiran < 0.7 else '✅ BAIK',
            'status_mahasiswa': status.upper(),
            'status_risk': '⚠️ BERISIKO' if status_risk else '✅ AMAN',
            'model_prob': f"{probability[1]:.2%}",
            'business_rule': 'DROPOUT' if actual_dropout else 'NON-DROPOUT'
        }
    }
    
    return result

def batch_predict(model, scaler, df_processed):
    """
    Predict for multiple students (REVISED)
    
    Parameters:
    -----------
    model : trained model
    scaler : fitted scaler
    df_processed : preprocessed dataframe
    
    Returns:
    --------
    predictions : list of risk levels
    dropout_probs : list of dropout probabilities
    """
    predictions = []
    dropout_probs = []
    risk_levels = []
    
    for _, row in df_processed.iterrows():
        try:
            result = predict_dropout_risk(
                model, scaler,
                row['IPK'], 
                row['Kehadiran'],
                row['Status']
            )
            predictions.append(result['prediction'])
            dropout_probs.append(result['dropout_probability'])
            risk_levels.append(result['risk_level'])
        except Exception as e:
            predictions.append('UNKNOWN')
            dropout_probs.append(0)
            risk_levels.append('UNKNOWN')
    
    return predictions, dropout_probs, risk_levels