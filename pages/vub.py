import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.getenv("OPENAI_API_KEY"))

def reload_df(conn):
    df = conn.read(worksheet="VUB")
    df["Periode"] = pd.to_datetime(df["Periode"]).dt.normalize()
    df.set_index("Periode", inplace=True)
    return df.sort_index()

@st.cache_resource
def load_model():
    return joblib.load("models/model_sarimax_vub.pkl")

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
        bulan_max_prediksi = pd.to_datetime(bulan_max) + pd.DateOffset(months=12)

        periode_2025 = pd.date_range(start="2025-01-01", end="2025-12-01", freq="MS")
        bulan_min_default = periode_2025.min()
        bulan_max_default = periode_2025.max()

        bulan_awal, bulan_akhir = st.slider(
            "Pilih rentang bulan:",
            min_value=bulan_min,
            max_value=bulan_max_prediksi.to_pydatetime(),
            value=(bulan_min_default.to_pydatetime(), bulan_max_default.to_pydatetime()),
            format="MM/YYYY"
        )

    st.subheader("📄 Data Aktual")
    st.dataframe(st.session_state.df_vub, use_container_width=True)

    df_filtered = st.session_state.df_vub.copy()
    
    df_filtered["Volume_lag1"] = df_filtered["Volume"].shift(1)
    df_filtered["Volume_lag2"] = df_filtered["Volume"].shift(2)
    df_filtered["Volume_lag3"] = df_filtered["Volume"].shift(3)
    df_filtered["Volume_lag6"] = df_filtered["Volume"].shift(6)
    df_filtered["Volume_lag12"] = df_filtered["Volume"].shift(12)
    df_filtered["Volume_roll_mean_3"] = df_filtered["Volume"].rolling(window=3).mean()
    df_filtered.fillna(method='bfill', inplace=True)

    try:
        model_fit = load_model()
        exog_features = ['BI Rate', 'APBN Infra', 'Effective Working Days', 'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag6', 'Volume_lag12', 'Volume_roll_mean_3']
        exog_hist = df_filtered[exog_features]

        last_period = df_filtered.index[-1]
        future_dates = pd.date_range(start=last_period + pd.DateOffset(months=1), periods=12, freq='MS')

        last_12_exog = exog_hist.iloc[-12:]
        exog_forecast = pd.concat([last_12_exog], ignore_index=True)
        exog_forecast.index = future_dates

        forecast = model_fit.forecast(steps=12, exog=exog_forecast)

        df_full_forecast = pd.DataFrame({
            "Volume": df_filtered["Volume"],
        })

        df_full_forecast = df_full_forecast.reindex(df_filtered.index.union(future_dates))
        df_full_forecast["Volume"].update(df_filtered["Volume"])
        df_full_forecast.loc[future_dates, "Volume"] = forecast.values

        st.success("✅ Prediksi 1 tahun ke depan berhasil dihitung!")

        filtered_df_full_forecast = df_full_forecast[(df_full_forecast.index >= bulan_awal) & (df_full_forecast.index <= bulan_akhir)]

        st.subheader("📈 Hasil Peramalan")
        st.dataframe(filtered_df_full_forecast, width=300)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df_full_forecast.index,
            y=filtered_df_full_forecast["Volume"],
            mode="lines+markers+text",
            name="Volume Prediksi",
            textposition="top center",
            line=dict(color="royalblue", width=3),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="Volume: %{y:.2f}<extra></extra>"
        ))
        fig.update_layout(
            xaxis_title="Periode",
            yaxis_title="Volume Prediksi",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(t=40, b=40, l=20, r=20),
            height=500,
            autosize=True
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Gagal memuat model SARIMAX atau menghitung prediksi: {e}")

    st.subheader("🧠 Rekomendasi Strategis")
    with st.spinner("Menghasilkan analisis dengan AI..."):
        insight = generate_insight_with_gpt(filtered_df_full_forecast)
        st.markdown(insight)

if __name__ == "__main__" or st.runtime.exists():
    show()
