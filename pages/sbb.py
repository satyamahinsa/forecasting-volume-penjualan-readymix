import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from tqdm import tqdm
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = st.secrets['openai']['api_key']
client = OpenAI(base_url="https://models.github.ai/inference", api_key=api_key)

def reload_df(conn, sheet_name):
    df = conn.read(worksheet=sheet_name)
    df["Periode"] = pd.to_datetime(df["Periode"]).dt.normalize()
    df.set_index("Periode", inplace=True)
    return df.sort_index()


def evaluate_forecast(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    accuracy = 100 - mape  # definisi sederhana: 100% - MAPE
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'sMAPE (%)': smape,
        'Accuracy (%)': accuracy
    }


@st.cache_resource
def load_model(df_existing):
    last_year = df_existing.index.max().year
    train_df = df_existing[(df_existing.index.year < last_year)]
    test_df = df_existing[(df_existing.index.year >= last_year)]
    
    model = pm.auto_arima(train_df['Volume'], m=12, seasonal=True, 
                        start_p=0, start_q=0, max_order=8, 
                        test='adf',error_action='ignore',
                        suppress_warnings=True, stepwise=True, trace=True)
    order = model.get_params()['order']
    seasonal_order = model.get_params()['seasonal_order']
    # Daftar kandidat fitur
    candidate_features = ['BI Rate', 'Inflasi', 'APBN Infra',
                        'Effective Working Days', 'PDB Konstruksi']

    # Tempat menyimpan hasil
    results = []

    # Loop kombinasi subset fitur
    for k in range(2, 5):  # ubah rentang jika perlu
        for subset in tqdm(list(itertools.combinations(candidate_features, k))):
            # 1. Buat model SARIMAX
            model = SARIMAX(train_df['Volume'],
                            exog=train_df[list(subset)],
                            order=order,
                            seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)

            # 2. Prediksi
            y_pred = model_fit.predict(start=test_df['Volume'].index[0],
                                    end=test_df['Volume'].index[-1],
                                    exog=test_df[list(subset)],
                                    dynamic=False)
            y_true = test_df['Volume'].values

            # 3. Hitung metrik akurasi & error
            metrics = evaluate_forecast(y_true, y_pred)

            # 4. Simpan hasil
            results.append({
                'features': subset,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE (%)': metrics['MAPE (%)'],
                'sMAPE (%)': metrics['sMAPE (%)'],
                'Accuracy (%)': metrics['Accuracy (%)']
            })

    # Konversi ke DataFrame
    result_df = pd.DataFrame(results)

    # Sorting berdasarkan Accuracy tertinggi
    best_by_accuracy = result_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)
    best_features = best_by_accuracy.iloc[0]['features']
    model = SARIMAX(df_existing['Volume'], exog=df_existing[list(best_features)], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    st.write(best_by_accuracy)
    return model_fit, list(best_features)
    # return joblib.load("models/model_sarimax_sbb_update.pkl")

def generate_insight_with_gpt(df_full_forecast):
    data_summary = df_full_forecast[["Forecasting"]].tail(12).to_string()
    prompt = f"""
    Berikut adalah hasil peramalan volume penjualan readymix unit SBB PT Semen Indonesia selama 12 bulan ke depan:

    {data_summary}

    Berikan analisis tren, insight penting, dan rekomendasi bisnis berbasis data tersebut yang berfokus pada anak perusahaan PT Semen Indonesia, yaitu PT Solusi Bangun Beton. Tulis dalam bahasa Indonesia yang formal dan ringkas.
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
    st.title("📊 Peramalan Volume Penjualan ReadyMix SBB")

    conn = st.connection("gsheets", type=GSheetsConnection)

    if "df_sbb" not in st.session_state:
        st.session_state.df_sbb = reload_df(conn, "SBB")
    
    if "df_forecasting_assumptions" not in st.session_state:
        st.session_state.df_forecasting_assumptions = reload_df(conn, "Forecasting SBB")

    df = st.session_state.df_sbb
    forecasting_assumptions = st.session_state.df_forecasting_assumptions
    st.write(forecasting_assumptions.index.min())

    with st.sidebar:
        st.markdown("🗓️ **Filter Hasil Prediksi**")
        combined_index = pd.date_range(
            start=df.index.min(),
            end=forecasting_assumptions.index.max(),
            freq='MS'
        )
        periode_full = combined_index
        bulan_min = periode_full.min()
        bulan_max = periode_full.max()

        # periode_2025 = pd.date_range(start=bulan_min, end=bulan_max, freq="MS")
        # bulan_min_default = periode_2025.min()
        # bulan_max_default = periode_2025.max()

        bulan_awal, bulan_akhir = st.slider(
            "Pilih rentang bulan:",
            min_value=bulan_min,
            max_value=bulan_max,
            value=(forecasting_assumptions.index.min().to_pydatetime(), forecasting_assumptions.index.max().to_pydatetime()),
            format="MM/YYYY"
        )


    try:
        model_fit, best_features = load_model(df)
        exog_df = forecasting_assumptions[best_features]
        
        forecast_12_months = model_fit.forecast(steps=12, exog=exog_df[:12])
        forecasting_final = pd.DataFrame({
            "Forecasting": forecast_12_months
        }, index=pd.date_range(start=forecasting_assumptions.index.min(), periods=12, freq='MS'))
        st.dataframe(forecasting_final)
        # st.session_state.df_forecasting_assumptions['Forecasting'] = forecasting_final
        # conn.update(worksheet="Forecasting SBB", data=st.session_state.df_forecasting_assumptions.reset_index())
        
        # lakukan filter pada df hingga volume aktual yang tidak memiliki 0
        df_filtered = df[(df.index >= bulan_awal) & (df.index <= bulan_akhir)]
        forecasting_existing = df_filtered[(df_filtered.index >= bulan_awal) & (df_filtered.index <= bulan_akhir)]['Forecasting']
        
        full_forecasting = pd.concat([
            forecasting_existing,
            forecasting_final
        ])

        st.subheader("📈 Hasil Peramalan")

        fig = go.Figure()
        # trace untuk Volume Prediksi (forecasting_final)
        fig.add_trace(go.Scatter(
            x=full_forecasting.index,
            y=full_forecasting["Forecasting"],
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
