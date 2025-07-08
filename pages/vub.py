import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = st.secrets['openai']['api_key']
client = OpenAI(base_url="https://models.github.ai/inference", api_key=api_key)

def reload_df(conn):
    df = conn.read(worksheet="VUB")
    df["Periode"] = pd.to_datetime(df["Periode"]).dt.normalize()
    df.set_index("Periode", inplace=True)
    return df.sort_index()

@st.cache_resource
def load_model():
    return joblib.load("models/model_sarimax_vub_update.pkl")

def generate_insight_with_gpt(df_full_forecast):
    data_summary = df_full_forecast[["Volume"]].tail(12).to_string()
    prompt = f"""
    Berikut adalah hasil peramalan volume penjualan readymix unit VUB PT Semen Indonesia selama 12 bulan ke depan:

    {data_summary}

    Berikan analisis tren, insight penting, dan rekomendasi bisnis berbasis data tersebut yang berfokus pada anak perusahaan PT Semen Indonesia, yaitu PT Varia Usaha Beton. Tulis dalam bahasa Indonesia yang formal dan ringkas.
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah analis data ahli yang memberikan insight dari data forecasting."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Gagal mendapatkan insight dari AI: {e}"

def show():
    st.title("📊 Peramalan Volume Penjualan ReadyMix VUB")

    conn = st.connection("gsheets", type=GSheetsConnection)

    if "df_vub" not in st.session_state:
        st.session_state.df_vub = reload_df(conn)

    df = st.session_state.df_vub

    # Sidebar filter
    with st.sidebar:
        st.markdown("🗓️ **Filter Hasil Prediksi**")
        periode_full = df.index.to_pydatetime()
        bulan_min = periode_full.min()
        bulan_max = periode_full.max()

        periode_2025 = pd.date_range(start="2025-01-01", end="2025-12-01", freq="MS")
        bulan_min_default = periode_2025.min()
        bulan_max_default = periode_2025.max()

        bulan_awal, bulan_akhir = st.slider(
            "Pilih rentang bulan:",
            min_value=bulan_min,
            max_value=bulan_max,
            value=(bulan_min_default.to_pydatetime(), bulan_max_default.to_pydatetime()),
            format="MM/YYYY"
        )

    df = st.session_state.df_vub.copy()

    try:
        model_fit = load_model()
        exog_features = ['BI Rate', 'APBN Infra', 'PDB Konstruksi']
        exog_hist = df[exog_features]
        
        forecast_12_months = model_fit.forecast(steps=12, exog=exog_hist[-12:])
        forecasting_final = pd.DataFrame({
            "Volume": forecast_12_months
        }, index=pd.date_range(start=periode_2025[0], periods=12, freq='MS'))
        
        # lakukan filter pada df hingga volume aktual yang tidak memiliki 0
        df_filtered = df[(df.index >= bulan_awal) & (df.index <= bulan_akhir) & (df["Volume"] > 0)] 
        forecasting_final = forecasting_final[(forecasting_final.index >= bulan_awal) & (forecasting_final.index <= bulan_akhir)]
        # filtered_df_full_forecast = forecasting_final[(forecasting_final.index >= bulan_awal) & (forecasting_final.index <= bulan_akhir)]

        st.subheader("📈 Hasil Peramalan")

        fig = go.Figure()
        # trace untuk Volume Prediksi (forecasting_final)
        fig.add_trace(go.Scatter(
            x=forecasting_final.index,
            y=forecasting_final["Volume"],
            mode="lines+markers+text",
            name="Volume Prediksi",
            textposition="top center",
            line=dict(color="royalblue", width=3),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="Volume Prediksi: %{y:.2f}<extra></extra>"
        ))

        # trace untuk Data Aktual (df_filtered)
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered["Volume"],
            mode="lines+markers+text",
            name="Volume Aktual",
            line=dict(color="firebrick", width=3),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="Volume Aktual: %{y:.2f}<extra></extra>"
        ))

        # layout & opsi
        fig.update_layout(
            xaxis_title="Periode",
            yaxis_title="Volume",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=40, b=40, l=20, r=20),
            height=500,
            autosize=True,
            legend=dict(title="Keterangan", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Gagal memuat model SARIMAX atau menghitung prediksi: {e}")

    st.subheader("🧠 Rekomendasi Strategis")
    with st.spinner("Menghasilkan analisis dengan AI..."):
        insight = generate_insight_with_gpt(forecasting_final)
        st.markdown(insight)

if __name__ == "__main__" or st.runtime.exists():
    show()
