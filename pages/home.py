import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config.chart_theme import apply_chart_theme

def show(df, df_processed):
    """Display home page"""
    st.title("🎓 Dashboard Prediksi Risiko Dropout Mahasiswa")
    st.markdown("### Sistem Prediksi Berbasis Machine Learning")
    
    df = df.copy()
    df['Angkatan_Display'] = df['Angkatan'] - 4
    
    st.markdown("""
    Dashboard ini dirancang untuk membantu institusi pendidikan dalam:
    - 📊 Memantau status dan performa mahasiswa
    - 🔍 Mengidentifikasi mahasiswa berisiko dropout
    - 📈 Menganalisis tren akademik secara menyeluruh
    - 🎯 Memberikan intervensi tepat sasaran berdasarkan data
    """)

    # ===============================
    # 📈 Statistik Utama
    # ===============================
    st.markdown("---")
    st.subheader("📈 Statistik Utama")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Mahasiswa", f"{len(df):,}")

    with col2:
        dropout_rate = (df_processed['Target'].sum() / len(df)) * 100
        st.metric(
            "Dropout Rate",
            f"{dropout_rate:.1f}%",
            delta=f"{df_processed['Target'].sum()} mahasiswa",
            delta_color="inverse"
        )

    with col3:
        avg_ipk = df['IPK'].mean()
        st.metric("Rata-rata IPK", f"{avg_ipk:.2f}",
                  help="IPK rata-rata seluruh mahasiswa")

    with col4:
        avg_kehadiran = df['Kehadiran'].mean() * 100
        st.metric("Rata-rata Kehadiran", f"{avg_kehadiran:.1f}%",
                  help="Persentase kehadiran rata-rata mahasiswa")

    st.markdown("---")

    # ===============================
    # 📊 Grafik Status Mahasiswa
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distribusi Status Mahasiswa")

        status_counts = df['Status'].value_counts()
        status_colors = {
            'AKTIF': '#4CAF50',
            'LULUS': '#2196F3',
            'CUTI': '#FF9800',
            'KELUAR': '#F44336',
            'NON AKTIF': '#9E9E9E',
            'REGISTRASI': '#00BCD4'
        }

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Status Mahasiswa",
            color=status_counts.index,
            color_discrete_map=status_colors
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'
        )

        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # 📌 Caption (Wajib Sesuai Revisi Dosen)
        st.caption("""
        Grafik ini menunjukkan proporsi status akademik mahasiswa. 
        Status AKTIF merupakan kelompok terbesar,
        sedangkan status CUTI dan KELUAR perlu diperhatikan 
        karena berpotensi meningkatkan risiko dropout.
        """)

        # 📄 Tabel Ringkasan
        status_df = status_counts.reset_index()
        status_df.columns = ['Status', 'Jumlah']
        status_df['Persentase'] = (status_df['Jumlah'] / status_df['Jumlah'].sum() * 100).round(1)

        st.dataframe(
            status_df.style.format({'Jumlah': '{:,}', 'Persentase': '{:.1f}%'}),
            use_container_width=True,
            hide_index=True
        )

    # ===============================
    # 📊 Grafik Program Studi
    # ===============================
    with col2:
        st.subheader("🎯 Distribusi Program Studi")

        prodi_counts = df['Prodi'].value_counts()

        fig = px.bar(
            x=prodi_counts.index,
            y=prodi_counts.values,
            labels={'x': 'Program Studi', 'y': 'Jumlah Mahasiswa'},
            title="Mahasiswa per Program Studi",
            color=prodi_counts.values,
            color_continuous_scale='Blues',
            text=prodi_counts.values
        )

        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate='<b>Prodi:</b> %{x}<br><b>Jumlah:</b> %{y:,}<extra></extra>'
        )

        fig.update_layout(showlegend=False)
        fig = apply_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # 📌 Caption Penjelasan
        st.caption("""
        Grafik ini menampilkan jumlah mahasiswa setiap program studi. 
        Program studi dengan jumlah mahasiswa lebih sedikit 
        dapat menjadi fokus analisis tambahan untuk melihat 
        potensi risiko dropout berdasarkan karakteristik prodi.
        """)

        # 📄 Tabel Ringkasan
        prodi_df = prodi_counts.reset_index()
        prodi_df.columns = ['Program Studi', 'Jumlah']
        prodi_df['Persentase'] = (prodi_df['Jumlah'] / prodi_df['Jumlah'].sum() * 100).round(1)

        st.dataframe(
            prodi_df.style.format({'Jumlah': '{:,}', 'Persentase': '{:.1f}%'}),
            use_container_width=True,
            hide_index=True
        )

    # ===============================
    # Footer
    # ===============================
    st.markdown("""
    ### 💡 Navigasi Dashboard

    Gunakan menu di sidebar untuk:
    - **📊 Dashboard Analitik** — Analisis IPK, Kehadiran, dan SKS.
    - **🔮 Prediksi Individu** — Prediksi risiko dropout secara personal.
    - **📈 Analisis Mahasiswa** — Identifikasi mahasiswa berisiko tinggi.
    - **ℹ️ Info Model** — Informasi model machine learning dan evaluasinya.
    """)

    st.success("**Dashboard siap digunakan!** Pilih menu di sidebar untuk memulai analisis.")
