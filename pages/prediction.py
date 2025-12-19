import streamlit as st
import plotly.graph_objects as go
from utils.predictor import predict_dropout_risk

def show(model, scaler):
    """Display individual prediction page (REVISED)"""
    st.title("🔮 Prediksi Risiko Dropout Individu")
    
    st.markdown("""
    Masukkan data mahasiswa untuk memprediksi risiko dropout.
    
    **Kriteria Dropout**: IPK < 2.0 **DAN** Kehadiran < 70%
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Data Mahasiswa")
        
        nim = st.text_input("NIM", placeholder="825160001")
        nama = st.text_input("Nama", placeholder="Nama Mahasiswa")
        
        prodi = st.selectbox("Program Studi", ["SI", "TI"])
        angkatan = st.selectbox("Angkatan", [2020, 2019, 2018, 2017, 2016])
        
        # Status selection
        status = st.selectbox(
            "Status Mahasiswa", 
            ["AKTIF", "LULUS", "CUTI", "KELUAR", "NON AKTIF", "REGISTRASI"]
        )
    
    with col2:
        st.subheader("📊 Data Akademik")
        
        ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)
        kehadiran = st.slider("Kehadiran (%)", 0, 100, 80, 1) / 100
        
        # Show warning if criteria met
        if ipk < 2.0 and kehadiran < 0.7:
            st.error("⚠️ **WARNING**: Memenuhi kriteria dropout!")
        elif ipk < 2.0:
            st.warning("⚠️ IPK di bawah standar minimum (2.0)")
        elif kehadiran < 0.7:
            st.warning("⚠️ Kehadiran di bawah 70%")
    
    st.markdown("---")
    
    if st.button("🔮 Prediksi Risiko", type="primary", use_container_width=True):
        with st.spinner("Memproses prediksi..."):
            result = predict_dropout_risk(
                model, scaler, 
                ipk, kehadiran, status
            )
            
            _display_prediction_result(result, nim, nama, prodi, angkatan)

def _display_prediction_result(result, nim, nama, prodi, angkatan):
    """Display prediction results (REVISED)"""
    st.success("✅ Prediksi Berhasil!")
    
    # Student info
    if nim and nama:
        st.info(f"**{nim}** | {nama} | {prodi} - {angkatan}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status Prediksi", result['prediction'])
    
    with col2:
        st.metric("Probabilitas Dropout", f"{result['dropout_probability']:.1%}")
    
    with col3:
        st.metric("Level Risiko", result['risk_level'])
    
    with col4:
        st.metric("Kondisi Aktual", 
                 "DROPOUT" if result['actual_dropout_condition'] else "NON-DROPOUT")
    
    # Risk Level Display
    st.markdown("### 📊 Detail Hasil Prediksi")
    
    if result['risk_level'] == 'TINGGI':
        st.markdown("""
        <div style='padding: 20px; background-color: #ffebee; border-left: 5px solid #c62828; border-radius: 5px;'>
        <h4 style='color: #000000; margin: 0;'>⚠️ RISIKO TINGGI</h4>
        <p style='color: #000000; margin: 5px 0 0 0;'>Memerlukan intervensi segera!</p>
        </div>
        """, unsafe_allow_html=True)
    elif result['risk_level'] == 'SEDANG':
        st.markdown("""
        <div style='padding: 20px; background-color: #fff3e0; border-left: 5px solid #ef6c00; border-radius: 5px;'>
        <h4 style='color: #000000; margin: 0;'>⚡ RISIKO SEDANG</h4>
        <p style='color: #000000; margin: 5px 0 0 0;'>Perlu monitoring intensif</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding: 20px; background-color: #e8f5e9; border-left: 5px solid #2e7d32; border-radius: 5px;'>
        <h4 style='color: #000000; margin: 0;'>✅ RISIKO RENDAH</h4>
        <p style='color: #000000; margin: 5px 0 0 0;'>Mahasiswa dalam kondisi baik</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed information
    st.markdown("### 📋 Informasi Detail")
    
    details = result['details']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **IPK**: {details['ipk']:.2f} {details['ipk_status']}
        
        **Kehadiran**: {details['kehadiran']} {details['kehadiran_status']}
        """)
    
    with col2:
        st.markdown(f"""
        **Status Mahasiswa**: {details['status_mahasiswa']} {details['status_risk']}
        
        **Business Rule**: {details['business_rule']}
        """)
    
    # Probability Gauge
    st.markdown("### 📊 Visualisasi Probabilitas")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['dropout_probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilitas Dropout (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'steps': [
                {'range': [0, 40], 'color': '#4caf50'},
                {'range': [40, 70], 'color': '#ff9800'},
                {'range': [70, 100], 'color': '#f44336'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Rekomendasi")
    
    if result['risk_level'] == 'TINGGI':
        st.error("""
        **Tindakan yang Disarankan (PRIORITAS TINGGI):**
        
        1. 🚨 **Konseling Akademik Segera**
           - Pertemuan dengan pembimbing akademik dalam 1-2 hari
           - Identifikasi akar masalah akademik
        
        2. 📚 **Program Remedial Intensif**
           - Tutoring untuk mata kuliah bermasalah
           - Study group dengan mahasiswa berprestasi
        
        3. 👨‍👩‍👦 **Melibatkan Orang Tua/Wali**
           - Komunikasi kondisi akademik
           - Dukungan dari keluarga
        
        4. 📊 **Evaluasi Beban Studi**
           - Pertimbangkan pengurangan SKS
           - Fokus pada mata kuliah wajib
        
        5. ⏰ **Monitoring Ketat**
           - Laporan mingguan ke pembimbing
           - Tracking kehadiran real-time
        """)
    elif result['risk_level'] == 'SEDANG':
        st.warning("""
        **Tindakan yang Disarankan (MONITORING):**
        
        1. 📅 **Konseling Berkala**
           - Pertemuan bi-weekly dengan pembimbing
           - Progress review setiap 2 minggu
        
        2. 💬 **Konseling Ringan**
           - Identifikasi kendala belajar
           - Time management coaching
        
        3. 📈 **Evaluasi Progress**
           - Monitor perkembangan IPK
           - Tracking kehadiran
        
        4. 🤝 **Peer Support**
           - Bergabung dengan study group
           - Mentoring dari senior
        """)
    else:
        st.success("""
        **Tindakan yang Disarankan (MAINTENANCE):**
        
        1. ✅ **Pertahankan Performa**
           - Konsistensi dalam belajar
           - Jaga kehadiran dan IPK
        
        2. 🎓 **Pengembangan Diri**
           - Ikut organisasi/lomba
           - Soft skills development
        
        3. 🌟 **Monitoring Rutin**
           - Check-in bulanan dengan pembimbing
           - Review progress semester
        
        4. 💼 **Persiapan Karir**
           - Magang/internship
           - Portfolio building
        """)
    
    # Additional insights
    with st.expander("📌 Penjelasan Hasil Prediksi"):
        st.markdown(f"""
        **Model Prediction Probability**: {details['model_prob']}
        
        **Business Rule**: 
        - Dropout = IPK < 2.0 **DAN** Kehadiran < 70%
        - Kondisi aktual: **{details['business_rule']}**
        
        **Logika Prediksi**:
        - Model menggunakan 3 features: IPK, Kehadiran, Status Risk
        - Threshold probability: 75% untuk klasifikasi dropout
        - Enhanced logic: Business rule + Model probability
        
        **Risk Level Determination**:
        - **TINGGI**: Dropout probability > 70% ATAU memenuhi business rule
        - **SEDANG**: Dropout probability 40-70%
        - **RENDAH**: Dropout probability < 40%
        """)