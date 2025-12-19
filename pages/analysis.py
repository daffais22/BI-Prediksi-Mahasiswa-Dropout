import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.predictor import batch_predict
from utils.preprocessor import process_data

def show(df, model, scaler):
    """Display student analysis page (REVISED)"""
    st.title("📈 Analisis Detail Mahasiswa")
    
    st.markdown("""
    Analisis komprehensif risiko dropout untuk seluruh mahasiswa.
    
    **Kriteria Dropout**: IPK < 2.0 **DAN** Kehadiran < 70%
    """)
    
    # Convert Angkatan (subtract 4 years for display)
    df = df.copy()
    df['Angkatan_Display'] = df['Angkatan'] - 4
    
    # Process and predict
    with st.spinner("Memproses prediksi untuk semua mahasiswa..."):
        df_processed = process_data(df)
        predictions, dropout_probs, risk_levels = batch_predict(model, scaler, df_processed)
        
        df_analysis = df.copy()
        df_analysis['Prediction'] = predictions
        df_analysis['Risk_Level'] = risk_levels
        df_analysis['Dropout_Probability'] = [p * 100 for p in dropout_probs]  # Convert to percentage
        
        # Add actual dropout condition
        df_analysis['Actual_Dropout'] = df_analysis.apply(
            lambda row: 'DROPOUT' if (row['Kehadiran'] < 0.7 and row['IPK'] < 2.0) else 'NON-DROPOUT',
            axis=1
        )
    
    # Summary metrics at top
    _display_summary_metrics(df_analysis)
    
    
    # Filters
    st.subheader("🔍 Filter Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_prodi = st.multiselect(
            "Program Studi",
            options=['Semua'] + list(df_analysis['Prodi'].unique()),
            default=['Semua']
        )
    
    with col2:
        filter_angkatan = st.multiselect(
            "Angkatan",
            options=['Semua'] + sorted(df_analysis['Angkatan_Display'].unique(), reverse=True),
            default=['Semua']
        )
    
    with col3:
        filter_status = st.multiselect(
            "Status",
            options=['Semua'] + list(df_analysis['Status'].unique()),
            default=['Semua']
        )
    
    with col4:
        filter_risk = st.multiselect(
            "Level Risiko",
            options=['Semua', 'TINGGI', 'SEDANG', 'RENDAH'],
            default=['Semua']
        )
    
    # Apply filters
    df_display = _apply_filters(df_analysis, filter_prodi, filter_angkatan, filter_status, filter_risk)
    
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Mahasiswa", 
        "📊 Visualisasi", 
        "📈 Statistik Detail",
        "🔴 High Risk Students"
    ])
    
    with tab1:
        _display_student_table(df_display)
    
    with tab2:
        _display_visualizations(df_display)
    
    with tab3:
        _display_detailed_statistics(df_display)
    
    with tab4:
        _display_high_risk_students(df_display)

def _display_summary_metrics(df_analysis):
    """Display summary metrics at the top"""
    st.subheader("📊 Ringkasan Keseluruhan")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(df_analysis)
    total_dropout_pred = len(df_analysis[df_analysis['Prediction'] == 'RISIKO DROPOUT'])
    total_actual_dropout = len(df_analysis[df_analysis['Actual_Dropout'] == 'DROPOUT'])
    high_risk = len(df_analysis[df_analysis['Risk_Level'] == 'TINGGI'])
    avg_prob = df_analysis['Dropout_Probability'].mean()
    
    with col1:
        st.metric(
            "Total Mahasiswa",
            f"{total:,}",
            help="Total mahasiswa dalam dataset"
        )
    
    with col2:
        dropout_rate = (total_dropout_pred / total * 100) if total > 0 else 0
        st.metric(
            "Predicted Dropout",
            f"{total_dropout_pred:,}",
            delta=f"{dropout_rate:.1f}%",
            delta_color="inverse",
            help="Mahasiswa yang diprediksi berisiko dropout"
        )
    
    with col3:
        actual_rate = (total_actual_dropout / total * 100) if total > 0 else 0
        st.metric(
            "Actual Dropout",
            f"{total_actual_dropout:,}",
            delta=f"{actual_rate:.1f}%",
            delta_color="inverse",
            help="Mahasiswa yang memenuhi kriteria dropout (IPK<2.0 & Kehadiran<70%)"
        )
    
    with col4:
        high_risk_rate = (high_risk / total * 100) if total > 0 else 0
        st.metric(
            "Risiko Tinggi",
            f"{high_risk:,}",
            delta=f"{high_risk_rate:.1f}%",
            delta_color="inverse",
            help="Mahasiswa dengan level risiko tinggi"
        )
    
    with col5:
        st.metric(
            "Avg Dropout Prob",
            f"{avg_prob:.1f}%",
            help="Rata-rata probabilitas dropout"
        )

def _apply_filters(df, filter_prodi, filter_angkatan, filter_status, filter_risk):
    """Apply filters to dataframe"""
    df_filtered = df.copy()
    
    if 'Semua' not in filter_prodi:
        df_filtered = df_filtered[df_filtered['Prodi'].isin(filter_prodi)]
    
    if 'Semua' not in filter_angkatan:
        df_filtered = df_filtered[df_filtered['Angkatan_Display'].isin(filter_angkatan)]
    
    if 'Semua' not in filter_status:
        df_filtered = df_filtered[df_filtered['Status'].isin(filter_status)]
    
    if 'Semua' not in filter_risk:
        df_filtered = df_filtered[df_filtered['Risk_Level'].isin(filter_risk)]
    
    return df_filtered

def _display_student_table(df_display):
    """Display student data table with styling"""
    st.subheader(f"📋 Daftar Mahasiswa ({len(df_display):,} mahasiswa)")
    
    df_display_styled = df_display[['NIM', 'Nama', 'Prodi', 'Angkatan_Display', 'Semester', 'Status', 'IPK', 'SKS', 'Kehadiran', 'Prediction', 'Risk_Level', 'Dropout_Probability', 'Actual_Dropout']].copy()
    
    df_display_styled['Kehadiran'] = (df_display_styled['Kehadiran'] * 100).round(1)
    df_display_styled['Dropout_Probability'] = df_display_styled['Dropout_Probability'].round(1)
    
    # Rename columns for better display
    df_display_styled.columns = [
        'NIM', 'Nama', 'Prodi', 'Angkatan', 'Semester', 'Status',
        'IPK', 'SKS', 'Kehadiran (%)', 'Prediksi', 'Level Risiko',
        'Prob. Dropout (%)', 'Kondisi Aktual'
    ]
    
    # Styling function
    def highlight_risk(row):
        if row['Level Risiko'] == 'TINGGI':
            return ['background-color: #ffebee'] * len(row)
        elif row['Level Risiko'] == 'SEDANG':
            return ['background-color: #fff3e0'] * len(row)
        elif row['Level Risiko'] == 'RENDAH':
            return ['background-color: #e8f5e9'] * len(row)
        return [''] * len(row)
    
    # Display styled dataframe
    st.dataframe(
        df_display_styled.style.apply(highlight_risk, axis=1)
        .format({
            'IPK': '{:.2f}',
            'Kehadiran (%)': '{:.1f}',
            'Prob. Dropout (%)': '{:.1f}'
        })
        .set_properties(**{'color': 'black'}),
        use_container_width=True,
        height=500
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_display_styled.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name='prediksi_dropout_mahasiswa.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        # Export only high risk students
        high_risk_df = df_display_styled[df_display_styled['Level Risiko'] == 'TINGGI']
        csv_high = high_risk_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🔴 Download High Risk Only (CSV)",
            data=csv_high,
            file_name='high_risk_students.csv',
            mime='text/csv',
            use_container_width=True
        )

def _display_visualizations(df_display):
    """Display visualizations"""
    st.subheader("📊 Visualisasi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Level Distribution
        risk_counts = df_display['Risk_Level'].value_counts()
        
        fig_risk = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                text=risk_counts.values,
                textposition='auto',
                marker_color=['#f44336', '#ff9800', '#4caf50'],
                hovertemplate='<b>Level Risiko:</b> %{x}<br><b>Jumlah Mahasiswa:</b> %{y}<extra></extra>'
            )
        ])
        
        fig_risk.update_layout(
            title="Distribusi Level Risiko",
            xaxis_title="Level Risiko",
            yaxis_title="Jumlah Mahasiswa",
            height=400
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Dropout Probability Distribution
        fig_prob = px.histogram(
            df_display,
            x='Dropout_Probability',
            nbins=50,
            title="Distribusi Probabilitas Dropout",
            labels={'Dropout_Probability': 'Probabilitas Dropout (%)'},
            color_discrete_sequence=['#2196F3']
        )
        
        fig_prob.add_vline(
            x=50, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text="Threshold 50%"
        )
        
        fig_prob.update_traces(
            hovertemplate='<b>Probabilitas:</b> %{x:.1f}%<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        
        fig_prob.update_layout(height=400)
        
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Risk by Prodi
    col3, col4 = st.columns(2)
    
    with col3:
        risk_prodi = pd.crosstab(df_display['Prodi'], df_display['Risk_Level'])
        
        fig_prodi = px.bar(
            risk_prodi,
            barmode='stack',
            title="Level Risiko per Program Studi",
            labels={'value': 'Jumlah', 'Prodi': 'Program Studi'},
            color_discrete_map={'TINGGI': '#f44336', 'SEDANG': '#ff9800', 'RENDAH': '#4caf50'}
        )
        
        fig_prodi.update_traces(
            hovertemplate='<b>Prodi:</b> %{x}<br><b>Level Risiko:</b> %{fullData.name}<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        
        fig_prodi.update_layout(height=400)
        
        st.plotly_chart(fig_prodi, use_container_width=True)
    
    with col4:
        risk_angkatan = pd.crosstab(df_display['Angkatan_Display'], df_display['Risk_Level'])
        
        fig_angkatan = px.bar(
            risk_angkatan,
            barmode='stack',
            title="Level Risiko per Angkatan",
            labels={'value': 'Jumlah', 'Angkatan_Display': 'Angkatan'},
            color_discrete_map={'TINGGI': '#f44336', 'SEDANG': '#ff9800', 'RENDAH': '#4caf50'}
        )
        
        fig_angkatan.update_layout(xaxis_title='Angkatan')
        fig_angkatan.update_traces(
            hovertemplate='<b>Angkatan:</b> %{x}<br><b>Level Risiko:</b> %{fullData.name}<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        
        fig_angkatan.update_layout(height=400)
        
        st.plotly_chart(fig_angkatan, use_container_width=True)
    
    # IPK vs Dropout Probability Scatter
    fig_scatter = px.scatter(
        df_display,
        x='IPK',
        y='Dropout_Probability',
        color='Risk_Level',
        size='Kehadiran',
            hover_data={
            'NIM': True,
            'Nama': True,
            'Prodi': True,
            'Angkatan_Display': True,
            'IPK': ':.2f',
            'Kehadiran': ':.2%',
            'Dropout_Probability': ':.1f',
            'Risk_Level': True
        },
        title="IPK vs Probabilitas Dropout",
        labels={
            'Dropout_Probability': 'Probabilitas Dropout (%)',
            'IPK': 'IPK',
            'Risk_Level': 'Level Risiko',
            'Kehadiran': 'Kehadiran',
            'Angkatan_Display': 'Angkatan'
        },
        color_discrete_map={'TINGGI': '#f44336', 'SEDANG': '#ff9800', 'RENDAH': '#4caf50'}
    )
    
    fig_scatter.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="red",
        line_width=2,
        annotation_text="Threshold Dropout 50%"
    )
    
    fig_scatter.add_vline(
        x=2.0, 
        line_dash="dash", 
        line_color="blue",
        line_width=2,
        annotation_text="IPK Threshold 2.0"
    )
    
    fig_scatter.update_traces(
        hovertemplate=(
            '<b>%{customdata[1]}</b><br>' +
            '<b>NIM:</b> %{customdata[0]}<br>' +
            '<b>Prodi:</b> %{customdata[2]}<br>' +
            '<b>Angkatan:</b> %{customdata[3]}<br>' +
            '<b>IPK:</b> %{x:.2f}<br>' +
            '<b>Dropout Prob:</b> %{y:.1f}%<br>' +
            '<b>Kehadiran:</b> %{marker.size:.1%}<br>' +
            '<b>Risk Level:</b> %{fullData.name}' +
            '<extra></extra>'
        )
    )
    
    fig_scatter.update_layout(height=500)
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def _display_detailed_statistics(df_display):
    """Display detailed statistics"""
    st.subheader("📈 Statistik Detail")
    
    # Risk Level Statistics
    st.markdown("#### 🎯 Statistik per Level Risiko")
    
    for level in ['TINGGI', 'SEDANG', 'RENDAH']:
        level_data = df_display[df_display['Risk_Level'] == level]
        
        if len(level_data) > 0:
            with st.expander(f"**{level}** ({len(level_data):,} mahasiswa)"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Jumlah", f"{len(level_data):,}")
                
                with col2:
                    st.metric("Avg IPK", f"{level_data['IPK'].mean():.2f}")
                
                with col3:
                    st.metric("Avg Kehadiran", f"{level_data['Kehadiran'].mean() * 100:.1f}%")
                
                with col4:
                    st.metric("Avg Prob", f"{level_data['Dropout_Probability'].mean():.1f}%")
                
                # Detail by Prodi
                st.markdown("**Distribusi per Prodi:**")
                prodi_counts = level_data['Prodi'].value_counts()
                for prodi, count in prodi_counts.items():
                    pct = count / len(level_data) * 100
                    st.write(f"- {prodi}: {count} mahasiswa ({pct:.1f}%)")
    
    st.markdown("---")
    
    # Status Statistics
    st.markdown("#### 📊 Statistik per Status Mahasiswa")
    
    status_risk = pd.crosstab(df_display['Status'], df_display['Risk_Level'])
    
    st.dataframe(status_risk, use_container_width=True)
    
    st.markdown("---")
    
    # Angkatan Summary
    st.markdown("#### 📅 Ringkasan per Angkatan")
    
    angkatan_summary = df_display.groupby('Angkatan_Display').agg({
        'NIM': 'count',
        'Dropout_Probability': 'mean',
        'IPK': 'mean',
        'Kehadiran': 'mean'
    }).round(2)
    
    angkatan_summary.columns = ['Total', 'Avg Dropout Prob (%)', 'Avg IPK', 'Avg Kehadiran']
    angkatan_summary['Avg Kehadiran'] = (angkatan_summary['Avg Kehadiran'] * 100).round(1)
    angkatan_summary.index.name = 'Angkatan'
    
    st.dataframe(angkatan_summary.sort_index(ascending=False), use_container_width=True)

def _display_high_risk_students(df_display):
    """Display high risk students with priority"""
    st.subheader("🔴 Mahasiswa Berisiko Tinggi - PRIORITAS INTERVENSI")
    
    high_risk = df_display[df_display['Risk_Level'] == 'TINGGI'].copy()
    
    if len(high_risk) == 0:
        st.success("✅ Tidak ada mahasiswa dengan risiko tinggi!")
        return
    
    st.warning(f"⚠️ **{len(high_risk):,} mahasiswa** memerlukan intervensi segera!")
    
    # Sort by dropout probability
    high_risk_sorted = high_risk.sort_values('Dropout_Probability', ascending=False)
    
    # Top 10 Critical
    st.markdown("### 🚨 TOP 10 MOST CRITICAL")
    
    top_10 = high_risk_sorted.head(10)
    
    for idx, (_, student) in enumerate(top_10.iterrows(), 1):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                st.markdown(f"""
                **#{idx} - {student['Nama']}**  
                📋 {student['NIM']} | {student['Prodi']} - {student['Angkatan_Display']}  
                📊 Status: {student['Status']}
                """)
            
            with col2:
                st.markdown(f"""
                **IPK:** {student['IPK']:.2f} {'❌' if student['IPK'] < 2.0 else '✅'}  
                **Kehadiran:** {student['Kehadiran']*100:.1f}% {'❌' if student['Kehadiran'] < 0.7 else '✅'}
                """)
            
            with col3:
                st.markdown(f"""
                **Prob Dropout:** {student['Dropout_Probability']:.1f}%  
                **Kondisi:** {student['Actual_Dropout']}
                """)
            
            st.markdown("---")
    
    # All High Risk Table
    st.markdown(f"### 📋 Semua Mahasiswa Risiko Tinggi ({len(high_risk):,})")
    
    high_risk_display = high_risk_sorted[[
        'NIM', 'Nama', 'Prodi', 'Angkatan_Display', 'Status',
        'IPK', 'Kehadiran', 'Dropout_Probability', 'Actual_Dropout'
    ]].copy()
    
    high_risk_display['Kehadiran'] = (high_risk_display['Kehadiran'] * 100).round(1)
    
    high_risk_display.columns = [
        'NIM', 'Nama', 'Prodi', 'Angkatan', 'Status',
        'IPK', 'Kehadiran (%)', 'Prob. Dropout (%)', 'Kondisi Aktual'
    ]
    
    st.dataframe(
        high_risk_display.style.format({
            'IPK': '{:.2f}',
            'Kehadiran (%)': '{:.1f}',
            'Prob. Dropout (%)': '{:.1f}'
        }).set_properties(**{'background-color': '#ffebee', 'color': 'black'}),
        use_container_width=True,
        height=400
    )
    
    # Action Plan
    st.markdown("### 💡 Rencana Tindakan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **Tindakan Segera (1-3 Hari):**
        
        1. 📞 **Kontak Mahasiswa**
           - Call/WA mahasiswa high risk
           - Jadwalkan meeting
        
        2. 📧 **Notifikasi Orang Tua**
           - Email kondisi akademik
           - Request dukungan keluarga
        
        3. 📋 **Assessment**
           - Identifikasi masalah utama
           - Buat action plan individual
        """)
    
    with col2:
        st.warning("""
        **Tindakan Lanjutan (1 Minggu):**
        
        1. 🎓 **Program Remedial**
           - Tutoring intensif
           - Study group
        
        2. 📊 **Monitoring Ketat**
           - Weekly report
           - Daily attendance check
        
        3. 🤝 **Support System**
           - Peer mentoring
           - Counseling akademik
        """)
    
    # Download high risk list
    csv_high = high_risk_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Daftar High Risk (CSV)",
        data=csv_high,
        file_name=f'high_risk_students_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        use_container_width=True
    )
