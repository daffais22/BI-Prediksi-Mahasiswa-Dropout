import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.predictor import batch_predict
from utils.preprocessor import process_data
from config.settings import RISK_COLORS

def show(df, model, scaler):
    """Display analytics dashboard"""
    st.title("📊 Dashboard Analitik")
    
    # Convert Angkatan (subtract 4 years for display)
    df = df.copy()
    df['Angkatan_Display'] = df['Angkatan'] - 4
    
    # Filters
    st.sidebar.subheader("🔍 Filter Data")
    selected_prodi = st.sidebar.multiselect(
        "Program Studi:",
        options=df['Prodi'].unique(),
        default=df['Prodi'].unique()
    )
    
    # Get unique display angkatan values
    angkatan_display_options = sorted(df['Angkatan_Display'].unique())
    selected_angkatan_display = st.sidebar.multiselect(
        "Angkatan:",
        options=angkatan_display_options,
        default=angkatan_display_options
    )
    
    # Convert selected display angkatan back to actual angkatan for filtering
    selected_angkatan = [a + 4 for a in selected_angkatan_display]
    
    # Filter data
    df_filtered = df[
        (df['Prodi'].isin(selected_prodi)) & 
        (df['Angkatan'].isin(selected_angkatan))
    ]
    df_filtered_processed = process_data(df_filtered)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tren IPK",
        "👥 Kehadiran",
        "📚 SKS",
        "🎯 Kategori Risiko"
    ])
    
    with tab1:
        _show_ipk_analysis(df_filtered)
    
    with tab2:
        _show_attendance_analysis(df_filtered)
    
    with tab3:
        _show_sks_analysis(df_filtered)
    
    with tab4:
        _show_risk_analysis(df_filtered_processed, model, scaler)

def _show_ipk_analysis(df_filtered):
    """Show IPK analysis"""
    st.subheader("📈 Analisis Tren IPK")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # IPK by Angkatan (using display angkatan)
        ipk_by_angkatan = df_filtered.groupby('Angkatan_Display')['IPK'].mean().reset_index()
        fig = px.line(
            ipk_by_angkatan,
            x='Angkatan_Display',
            y='IPK',
            title='Rata-rata IPK per Angkatan',
            markers=True,
            line_shape='spline'
        )
        fig.update_layout(
            yaxis_range=[0, 4],
            xaxis_title='Angkatan'
        )
        fig.update_traces(
            line={'color': '#2196F3', 'width': 3}, 
            marker={
                'size': 10, 
                'color': '#2196F3', 
                'line': {'color': '#ffffff', 'width': 2}
            },
            hovertemplate='<b>Angkatan:</b> %{x}<br><b>Rata-rata IPK:</b> %{y:.2f}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IPK Distribution
        fig = px.histogram(
            df_filtered,
            x='IPK',
            nbins=30,
            title='Distribusi IPK',
            labels={'IPK': 'IPK', 'count': 'Frekuensi'},
            color_discrete_sequence=['#4CAF50']
        )
        fig.add_vline(
            x=df_filtered['IPK'].mean(),
            line_dash="dash",
            line_color="#FF5722",
            line_width=2,
            annotation_text=f"Mean: {df_filtered['IPK'].mean():.2f}",
            annotation_font_size=12
        )
        fig.update_traces(
            hovertemplate='<b>IPK:</b> %{x:.2f}<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # IPK by Status
    st.subheader("IPK Berdasarkan Status")
    fig = px.box(
        df_filtered,
        x='Status',
        y='IPK',
        color='Status',
        title='Distribusi IPK per Status',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(
        hovertemplate='<b>Status:</b> %{x}<br><b>IPK:</b> %{y:.2f}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)

def _show_attendance_analysis(df_filtered):
    """Show attendance analysis"""
    st.subheader("👥 Analisis Kehadiran")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Kehadiran by Angkatan (using display angkatan)
        kehadiran_by_angkatan = df_filtered.groupby('Angkatan_Display')['Kehadiran'].mean().reset_index()
        kehadiran_by_angkatan['Kehadiran'] *= 100
        fig = px.bar(
            kehadiran_by_angkatan,
            x='Angkatan_Display',
            y='Kehadiran',
            title='Rata-rata Kehadiran per Angkatan (%)',
            color='Kehadiran',
            color_continuous_scale='RdYlGn',
            text='Kehadiran'
        )
        fig.update_layout(xaxis_title='Angkatan')
        fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='outside',
            hovertemplate='<b>Angkatan:</b> %{x}<br><b>Kehadiran:</b> %{y:.1f}%<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Kehadiran Distribution
        df_filtered_temp = df_filtered.copy()
        df_filtered_temp['Kehadiran_Pct'] = df_filtered_temp['Kehadiran'] * 100
        fig = px.histogram(
            df_filtered_temp,
            x='Kehadiran_Pct',
            nbins=30,
            title='Distribusi Kehadiran (%)',
            labels={'Kehadiran_Pct': 'Kehadiran (%)', 'count': 'Frekuensi'},
            color_discrete_sequence=['#FF9800']
        )
        fig.update_traces(
            hovertemplate='<b>Kehadiran:</b> %{x:.1f}%<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Kehadiran by Status
    st.subheader("Kehadiran Berdasarkan Status")
    fig = px.box(
        df_filtered,
        x='Status',
        y='Kehadiran',
        color='Status',
        title='Distribusi Kehadiran per Status',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        hovertemplate='<b>Status:</b> %{x}<br><b>Kehadiran:</b> %{y:.2%}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)

def _show_sks_analysis(df_filtered):
    """Show SKS analysis"""
    st.subheader("📚 Analisis SKS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SKS by Angkatan (using display angkatan)
        sks_by_angkatan = df_filtered.groupby('Angkatan_Display')['SKS'].mean().reset_index()
        fig = px.bar(
            sks_by_angkatan,
            x='Angkatan_Display',
            y='SKS',
            title='Rata-rata SKS per Angkatan',
            color='SKS',
            color_continuous_scale='Blues',
            text='SKS'
        )
        fig.update_layout(xaxis_title='Angkatan')
        fig.update_traces(
            texttemplate='%{text:.1f}', 
            textposition='outside',
            hovertemplate='<b>Angkatan:</b> %{x}<br><b>Rata-rata SKS:</b> %{y:.1f}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # SKS Distribution
        fig = px.histogram(
            df_filtered,
            x='SKS',
            nbins=30,
            title='Distribusi SKS',
            labels={'SKS': 'SKS', 'count': 'Frekuensi'},
            color_discrete_sequence=['#9C27B0']
        )
        fig.update_traces(
            hovertemplate='<b>SKS:</b> %{x}<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # SKS by Status
    st.subheader("SKS Berdasarkan Status")
    fig = px.box(
        df_filtered,
        x='Status',
        y='SKS',
        color='Status',
        title='Distribusi SKS per Status',
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(
        hovertemplate='<b>Status:</b> %{x}<br><b>SKS:</b> %{y}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)

def _show_risk_analysis(df_filtered_processed, model, scaler):
    """Show risk category analysis"""
    st.subheader("🎯 Kategori Risiko Mahasiswa")
    
    # Predict for filtered students
    predictions, dropout_probs, risk_levels = batch_predict(model, scaler, df_filtered_processed)
    df_filtered_processed['Prediction'] = predictions
    df_filtered_processed['Risk_Level'] = risk_levels
    df_filtered_processed['Dropout_Probability'] = [p * 100 for p in dropout_probs]
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = df_filtered_processed['Risk_Level'].value_counts()
        
        # Define colors for risk levels
        risk_color_map = {
            'TINGGI': '#f44336',
            'SEDANG': '#ff9800', 
            'RENDAH': '#4caf50'
        }
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Distribusi Kategori Risiko',
            color=risk_counts.index,
            color_discrete_map=risk_color_map
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by Prodi
        risk_by_prodi = df_filtered_processed.groupby(['Prodi', 'Risk_Level']).size().reset_index(name='Count')
        fig = px.bar(
            risk_by_prodi,
            x='Prodi',
            y='Count',
            color='Risk_Level',
            title='Kategori Risiko per Program Studi',
            barmode='group',
            color_discrete_map=risk_color_map,
            text='Count'
        )
        fig.update_traces(
            texttemplate='%{text}', 
            textposition='outside',
            hovertemplate='<b>Prodi:</b> %{x}<br><b>Risk Level:</b> %{fullData.name}<br><b>Jumlah:</b> %{y}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk by Angkatan
    st.subheader("Risiko per Angkatan")
    risk_by_angkatan = df_filtered_processed.groupby(['Angkatan_Display', 'Risk_Level']).size().reset_index(name='Count')
    fig = px.bar(
        risk_by_angkatan,
        x='Angkatan_Display',
        y='Count',
        color='Risk_Level',
        title='Kategori Risiko per Angkatan',
        barmode='stack',
        color_discrete_map=risk_color_map
    )
    fig.update_layout(xaxis_title='Angkatan')
    fig.update_traces(
        hovertemplate='<b>Angkatan:</b> %{x}<br><b>Risk Level:</b> %{fullData.name}<br><b>Jumlah:</b> %{y}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary Statistics
    st.markdown("---")
    st.subheader("📊 Statistik Risiko")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_count = len(df_filtered_processed[df_filtered_processed['Risk_Level'] == 'TINGGI'])
        high_risk_pct = (high_risk_count / len(df_filtered_processed) * 100) if len(df_filtered_processed) > 0 else 0
        st.metric(
            "Risiko Tinggi",
            f"{high_risk_count}",
            delta=f"{high_risk_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        medium_risk_count = len(df_filtered_processed[df_filtered_processed['Risk_Level'] == 'SEDANG'])
        medium_risk_pct = (medium_risk_count / len(df_filtered_processed) * 100) if len(df_filtered_processed) > 0 else 0
        st.metric(
            "Risiko Sedang",
            f"{medium_risk_count}",
            delta=f"{medium_risk_pct:.1f}%"
        )
    
    with col3:
        low_risk_count = len(df_filtered_processed[df_filtered_processed['Risk_Level'] == 'RENDAH'])
        low_risk_pct = (low_risk_count / len(df_filtered_processed) * 100) if len(df_filtered_processed) > 0 else 0
        st.metric(
            "Risiko Rendah",
            f"{low_risk_count}",
            delta=f"{low_risk_pct:.1f}%",
            delta_color="normal"
        )
    
    # Show detailed statistics per risk level
    st.markdown("---")
    st.subheader("📈 Detail Statistik per Level Risiko")
    
    for level in ['TINGGI', 'SEDANG', 'RENDAH']:
        level_data = df_filtered_processed[df_filtered_processed['Risk_Level'] == level]
        
        if len(level_data) > 0:
            with st.expander(f"**{level}** ({len(level_data)} mahasiswa)"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg IPK", f"{level_data['IPK'].mean():.2f}")
                
                with col2:
                    st.metric("Avg Kehadiran", f"{level_data['Kehadiran'].mean() * 100:.1f}%")
                
                with col3:
                    st.metric("Avg Prob", f"{level_data['Dropout_Probability'].mean():.1f}%")
                
                with col4:
                    st.metric("Count", f"{len(level_data)}")
