import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from tqdm import tqdm
import html
from bs4 import BeautifulSoup
import json, re
import holidays
from datetime import date, timedelta
import time


conn = st.connection("gsheets", type=GSheetsConnection)

def reload_df(conn, sheet_name):
    df = conn.read(worksheet=sheet_name)
    df["Periode"] = pd.to_datetime(df["Periode"])
    df.set_index("Periode", inplace=True)
    return df.sort_index()

def update_df_to_gsheet(df, sheet_name="VUB"):
    conn.update(worksheet=sheet_name, data=df.reset_index())

def get_effective_working_days(year, month):
    indo_holidays = holidays.country_holidays('ID', years=[year])
    start_date = date(year, month, 1)
    end_date = pd.Period(f"{year}-{month:02}").end_time.date()
    current = start_date
    workdays = 0
    while current <= end_date:
        if current.weekday() < 5 and current not in indo_holidays:
            workdays += 1
        current += timedelta(days=1)
    return workdays

def update_dataframe(df, updates_dict):
    """Update dataframe utama dengan hasil gabungan."""
    for col_name, series_update in updates_dict.items():
        df.loc[series_update.index, col_name] = series_update.values
    return df

def scrape_inflasi():
    API_KEY = st.secrets['scraping']['api_key']
    url = f"https://webapi.bps.go.id/v1/api/view/domain/0000/model/statictable/lang/ind/id/915/key/{API_KEY}"
    response = requests.get(url)
    json_data = response.json()
    html_encoded = json_data["data"]["table"]
    html_decoded = html.unescape(html_encoded)
    soup = BeautifulSoup(html_decoded, "html.parser")
    rows = soup.find_all('tr')
    data_months, data_years, data_inflation = [], [], []
    for row in rows:
        months = row.find_all('td', class_='xl6622202')
        months = [col.get_text(strip=True) for col in months]
        if months:
            data_months.append(months[0].replace('\xa0', '').strip())
    for row in rows:
        years = row.find_all('td', class_='xl7022202')
        years = [col.get_text(strip=True) for col in years]
        if years:
            data_years = years
    for row in rows:
        values = row.find_all('td', class_=['xl7222202', 'xl7122202'])
        values = [col.get_text(strip=True) for col in values]
        if values:
            data_inflation.append(values)
    inflation_data = []
    for i in range(len(data_months)):
        for j in range(len(data_years)):
            inflation_data.append({
                'Tahun': data_years[j],
                'Bulan': data_months[i],
                'Inflasi': data_inflation[i][j] if j < len(data_inflation[i]) else None
            })
    df_inflation = pd.DataFrame(inflation_data)
    df_inflation['Inflasi'] = df_inflation['Inflasi'].str.replace(',', '.', regex=False)
    df_inflation['Inflasi'] = pd.to_numeric(df_inflation['Inflasi'], errors='coerce')
    df_inflation = df_inflation.dropna(subset=['Inflasi']).reset_index(drop=True)
    df_inflation['Inflasi'] = df_inflation['Inflasi'] / 100
    df_inflation['Tahun'] = df_inflation['Tahun'].astype(int)
    month_map = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
        "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
    }
    df_inflation["Bulan"] = df_inflation["Bulan"].map(month_map)
    df_inflation['Periode'] = pd.to_datetime({
        'year': df_inflation['Tahun'],
        'month': df_inflation['Bulan'],
        'day': 1
    })
    df_inflation = df_inflation.set_index('Periode').drop(columns=['Tahun', 'Bulan']).sort_index()
    return df_inflation


def scrape_bi_rate():
    API_KEY = st.secrets['scraping']['api_key']
    url = f'https://webapi.bps.go.id/v1/api/list/model/data/lang/ind/domain/0000/var/379/key/{API_KEY}?th=2020-2025'

    response = requests.get(url)
    data = response.json()
    datacontent = data.get('datacontent', {})

    data_list = []

    for kode, value in datacontent.items():
        timecode = kode[6:]  # Ambil bagian akhir kode waktu

        try:
            if len(timecode) == 3:
                bulan = int(timecode[2])
                tahun = 2000 + int(timecode[:2])
            elif len(timecode) == 4:
                bulan = int(timecode[2:])
                tahun = 2000 + int(timecode[:2])
            else:
                continue
        except ValueError:
            continue

        data_list.append({
            'Tahun': tahun,
            'Bulan': bulan,
            'BI Rate': float(value)
        })

    df_bi_rate = pd.DataFrame(data_list)
    df_bi_rate = df_bi_rate.sort_values(by=['Tahun', 'Bulan']).reset_index(drop=True)
    df_bi_rate = df_bi_rate[df_bi_rate['Bulan'] <= 12]
    df_bi_rate = df_bi_rate[df_bi_rate['Tahun'] >= 2009]
    df_bi_rate['Periode'] = pd.to_datetime({
        'year': df_bi_rate['Tahun'],
        'month': df_bi_rate['Bulan'],
        'day': 1
    })
    df_bi_rate = df_bi_rate.set_index('Periode')
    df_bi_rate.drop(columns=['Tahun', 'Bulan'], inplace=True)
    df_bi_rate['BI Rate'] = df_bi_rate['BI Rate'] / 100
    return df_bi_rate


def scrape_apbn_infra():
    # ——— Ambil APBN Infrastruktur dari Kemenkeu ———
    url = "https://media.kemenkeu.go.id/SinglePage/custompage?p=/Pages/Home/Anggaran-Infrastruktur"
    data = json.loads(requests.get(url).text)['Data']['Content']

    df_apbn = pd.DataFrame({
        'Tahun': [int(item['Tahun']) for item in data],
        'APBN Infrastruktur': [
            float(item['Jumlah'].replace(',', '.')) for item in data
        ]
    })

    # Tambahkan data 2025 dari berita bisnis.com
    url_apbn_2025 = 'https://ekonomi.bisnis.com/read/20240816/45/1791651/anggaran-infrastruktur-rp400-triliun-untuk-proyek-prioritas-di-2025-apa-saja'
    response = requests.get(url_apbn_2025)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find('article').find('p').get_text(strip=True)

    tahun_2025_match = re.search(r'infrastruktur\s*(\d{4})', text)
    anggaran_2025_match = re.search(r'Rp\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)\s*triliun', text)

    if tahun_2025_match and anggaran_2025_match:
        tahun_2025 = int(tahun_2025_match.group(1))
        anggaran_2025 = float(
            anggaran_2025_match.group(1).replace('.', '').replace(',', '.')
        )
        df_apbn.loc[len(df_apbn)] = [tahun_2025, anggaran_2025]

    df_apbn = df_apbn.sort_values('Tahun').reset_index(drop=True)

    df_existing = st.session_state.df_vub
    
    # ——— Hitung data aktual bulanan ———
    periode_terakhir = df_existing.index.max()
    tahun_terakhir = periode_terakhir.year

    result = []

    for idx, row in df_existing.iterrows():
        year = idx.year
        month = idx.month
        volume = row['Volume']

        # ambil APBN tahunan
        apbn_value = df_apbn[df_apbn['Tahun'] == year]['APBN Infrastruktur'].values[0]
        # hitung proporsi bulan thd total tahun
        comparison_year = tahun_terakhir - 1 if year == tahun_terakhir else year
        total_volume = df_existing[df_existing.index.year == comparison_year]['Volume'].sum()
        ratio = volume / total_volume if total_volume > 0 else 0
        apbn_per_month = apbn_value * ratio

        result.append({
            'Tahun': year,
            'Bulan': month,
            'APBN Infra': apbn_per_month,
        })

    df_result = pd.DataFrame(result)
    df_result = df_result.sort_values(['Tahun', 'Bulan']).reset_index(drop=True)

    df_result['Periode'] = pd.to_datetime(
        df_result[['Tahun', 'Bulan']].rename(columns={'Tahun': 'year', 'Bulan': 'month'}).assign(day=1)
    )
    df_result.set_index('Periode', inplace=True)
    df_result.drop(columns=['Tahun', 'Bulan'], inplace=True)
    

    return df_result


def forecast_apbn_infra(df_actual):
    # ——— Hitung prediksi 12 bulan ke depan ———
    forecast_periods = pd.date_range(
        start=df_actual.index.max() + pd.DateOffset(months=1),
        periods=13, freq='MS'
    )

    df_result = df_actual.copy()

    # buat list untuk menyimpan baris hasil forecast
    forecast_rows = []

    for periode in forecast_periods:
        year = periode.year
        month = periode.month

        # fallback ke tahun-1 atau tahun-2
        apbn_per_month = None
        for offset in [1, 2]:
            prev_year = year - offset
            mask = (df_result.index.year == prev_year) & (df_result.index.month == month)
            if mask.any():
                apbn_per_month = df_result.loc[mask, 'APBN Infra'].values[0]
                break

        # jika tetap None, misalnya data sangat terbatas, default-kan ke 0
        if apbn_per_month is None:
            apbn_per_month = 0.0

        forecast_rows.append({
            'Tahun': year,
            'Bulan': month,
            'APBN Infra': apbn_per_month,
        })

    # ——— Gabungkan hasil lama + forecast ———
    df_forecast = pd.DataFrame(forecast_rows)
    df_combined = pd.concat([
        df_result.reset_index().rename(columns={'Periode': 'Periode'}),
        df_forecast
    ], ignore_index=True)

    # ——— Finalisasi index & kolom ———
    df_combined['Periode'] = pd.to_datetime(
        df_combined[['Tahun', 'Bulan']].rename(columns={'Tahun': 'year', 'Bulan': 'month'}).assign(day=1)
    )
    df_combined.set_index('Periode', inplace=True)
    df_combined.drop(columns=['Tahun', 'Bulan'], inplace=True)

    # urutkan indeks
    df_combined.sort_index(inplace=True)

    return df_combined


def scrape_pdb_konstruksi():
    return df[['PDB Konstruksi']]


def scrape_effective_working_days():
    df_existing = st.session_state.df_vub
    periods_actual = df_existing.index

    data = []

    for idx in periods_actual:
        ewd = get_effective_working_days(idx.year, idx.month)
        data.append({
            'Periode': idx,
            'Effective Working Days': ewd,
            'Sumber': 'Aktual'
        })

    df_ewd = pd.DataFrame(data).set_index('Periode').sort_index()
    df_ewd = df_ewd[['Effective Working Days']]

    return df_ewd

def forecast_effective_working_days(df_actual):
    df_existing = df_actual
    last_periode = df_existing.index.max()
    
    data = []

    forecast_periods = pd.date_range(
        start=last_periode + pd.DateOffset(months=1),
        periods=13,
        freq='MS'
    )

    for idx in forecast_periods:
        ewd = get_effective_working_days(idx.year, idx.month)
        data.append({
            'Periode': idx,
            'Effective Working Days': ewd,
        })
    
    df_ewd = pd.DataFrame(data).set_index('Periode').sort_index()
    df_ewd = df_ewd[['Effective Working Days']]

    return df_ewd


def sarimax_forecast(train_series, steps=13):
    model = pm.auto_arima(
        train_series, seasonal=True, m=12,
        trace=False, error_action='ignore', suppress_warnings=True
    )
    order = model.get_params()['order']
    seasonal_order = model.get_params()['seasonal_order']

    fitted = SARIMAX(train_series, order=order, seasonal_order=seasonal_order).fit()
    forecast = fitted.forecast(steps=steps)

    forecast_index = pd.date_range(
        start=train_series.index.max() + pd.DateOffset(months=1),
        periods=steps, freq='MS'
    )
    return pd.DataFrame({train_series.name: forecast.values}, index=forecast_index)


def update_or_forecast_column(col_name, df_existing, df_scraped, df_forecast, global_latest_index):
    df = df_existing.copy()
    combined_index = pd.date_range(
        start=min(df_scraped.index.min(), df_forecast.index.min()),
        end=global_latest_index,
        freq='MS'
    )

    # kerangka kosong
    combined = pd.DataFrame(index=combined_index, columns=[col_name], dtype=float)

    for idx in combined_index:
        current_val = df.loc[idx, col_name] if idx in df.index else None
        scraped_val = df_scraped.loc[idx, col_name] if idx in df_scraped.index else None
        forecast_val = df_forecast.loc[idx, col_name] if idx in df_forecast.index else None

        if current_val is not None and not pd.isna(current_val):
            # sudah ada nilai valid di df_existing → pakai itu
            combined.loc[idx, col_name] = current_val
        elif scraped_val is not None and not pd.isna(scraped_val):
            # kalau scraping terbaru ada & valid → isi
            combined.loc[idx, col_name] = scraped_val
        elif forecast_val is not None and not pd.isna(forecast_val):
            # kalau tetap kosong → pakai forecast
            combined.loc[idx, col_name] = forecast_val
        # kalau forecast juga None → biarkan NaN

    updated_actual = combined.loc[combined.index <= global_latest_index]
    forecast_df = df_forecast.loc[df_forecast.index > global_latest_index, [col_name]]

    return updated_actual, forecast_df


def process_all_columns(df_existing, scraped_data_dict, sheet_updater, update_forecasting, start_year=2020):
    updated_actuals = df_existing.copy()
    forecast_assumptions = pd.DataFrame(columns=['Periode'])

    # cari global_latest_index dari semua hasil scraping
    all_latest_indices = []
    for col, df_scraped in scraped_data_dict.items():
        if not df_scraped.empty:
            all_latest_indices.append(df_scraped.index.max())

    all_latest_indices.append(updated_actuals.index.max())
    global_latest_index = max(all_latest_indices)

    all_index = pd.date_range(
        start=min(updated_actuals.index.min(), global_latest_index),
        end=global_latest_index,
        freq='MS'
    )
    updated_actuals = updated_actuals.reindex(all_index)
    updated_actuals.index.name = "Periode"
    updated_actuals['Tahun'] = updated_actuals.index.year
    updated_actuals['Bulan'] = updated_actuals.index.month
    
    forecast_columns = ["BI Rate", "Inflasi", "PDB Konstruksi"]
    scraped_only_columns = ["APBN Infra", "Effective Working Days"]

    for col in forecast_columns:
        df_col = df_existing[[col]]
        df_col = df_col[df_col.index.year >= start_year]

        forecast_df = sarimax_forecast(df_col[col])
        scraped_df = scraped_data_dict.get(col, pd.DataFrame())
        scraped_df = scraped_df[scraped_df.index.year >= start_year]
        
        actual_df, forecast_df_col = update_or_forecast_column(
            col, df_col, scraped_df, forecast_df, global_latest_index
        )
        
        for c in actual_df:
            updated_actuals.loc[actual_df.index, c] = actual_df[c].values

        forecast_assumptions['Periode'] = forecast_df_col.index
        forecast_assumptions[col] = forecast_df_col.values

    # Kolom yang hanya scraped saja
    for col in scraped_only_columns:
        df_col = df_existing[[col]]
        df_col = df_col[df_col.index.year >= start_year]
        
        scraped_df = scraped_data_dict.get(col, pd.DataFrame())
        if col == "APBN Infra":
            forecast_df = forecast_apbn_infra(scraped_df)
        elif col == "Effective Working Days":
            forecast_df = forecast_effective_working_days(scraped_df)
        
        actual_df, forecast_df_col = update_or_forecast_column(
            col, df_col, scraped_df, forecast_df, global_latest_index
        )
        
        for c in actual_df:
            updated_actuals.loc[actual_df.index, c] = actual_df[c].values

        forecast_assumptions['Periode'] = forecast_df_col.index
        forecast_assumptions[col] = forecast_df_col.values
    
    forecast_assumptions.set_index('Periode', inplace=True)
    if update_forecasting:
        sheet_updater(forecast_assumptions, sheet_name="Forecasting VUB")

    return updated_actuals

def data_scraping(df, forecasting_assumptions, sheet_updater, update_forecasting, start_year=2020):
    prev_forecasting = forecasting_assumptions.copy()
    prev_forecasting = prev_forecasting['Forecasting']
    
    scraped_data_dict = {
        "BI Rate": scrape_bi_rate(),
        "Inflasi": scrape_inflasi(),
        "APBN Infra": scrape_apbn_infra(),
        "PDB Konstruksi": scrape_pdb_konstruksi(),
        "Effective Working Days": scrape_effective_working_days()
    }

    updated_actuals = process_all_columns(
        df,
        scraped_data_dict,
        sheet_updater=update_df_to_gsheet,
        update_forecasting=update_forecasting,
        start_year=start_year
    )
    
    updated_actuals['Volume'] = updated_actuals['Volume'].fillna(0)
    
    mask = prev_forecasting.index.isin(updated_actuals.index)
    for idx in prev_forecasting.index[mask]:
        updated_actuals.at[idx, 'Forecasting'] = prev_forecasting.at[idx]
    
    sheet_updater(updated_actuals, sheet_name="VUB")
    return updated_actuals


st.title("⚙️ Pengaturan Data VUB")

if "df_vub" not in st.session_state:
    st.session_state.df_vub = reload_df(conn, "VUB")

if "df_forecasting_assumptions" not in st.session_state:
    st.session_state.df_forecasting_assumptions = reload_df(conn, "Forecasting VUB")

df = st.session_state.df_vub
forecasting_assumptions = st.session_state.df_forecasting_assumptions

st.dataframe(df)

# --- Update Data Otomatis ---
with st.expander("🔄 Update Data Otomatis", expanded=True):
    update_forecasting = st.checkbox(
        "Perbarui data asumsi",
        value=False,
        help="Hilangkan centang jika tidak ingin memperbarui data asumsi."
    )

    if st.button("Ambil Data dari API", type="primary"):
        with st.spinner("Mengambil dan memproses data..."):
            try:
                update_actuals = data_scraping(df, forecasting_assumptions, update_df_to_gsheet, update_forecasting, 2020)
                st.session_state.df_vub = update_actuals
                st.toast("Data berhasil diperbarui!", icon="✅")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.toast(f"Gagal mengambil data: {e}", icon="❌")

col1, col2, col3 = st.columns(3)

# --- Input Data Baru ---
with col1:
    with st.expander("➕ Input Data Baru", expanded=True):
        last_periode = df.index.max() if not df.empty else pd.Timestamp.today()
        default_periode = (last_periode + pd.offsets.MonthBegin(1)).replace(day=1)

        periode = st.date_input("Periode", value=default_periode, format="YYYY-MM-DD", key="input_periode")
        periode = pd.to_datetime(periode)
        bi_rate = st.number_input("BI Rate", value=0.0, format="%.5f", key="input_bi_rate")
        inflasi = st.number_input("Inflasi", value=0.0, format="%.5f", key="input_inflasi")
        apbn_infra = st.number_input("APBN Infrastruktur", value=0.0, format="%.5f", key="input_apbn_infra")
        pdb_konstruksi = st.number_input("PDB Konstruksi", value=0.0, format="%.5f", key="input_pdb_konstruksi")
        ewd = st.number_input("Hari Kerja Efektif", min_value=1, max_value=31,
                                value=get_effective_working_days(default_periode.year, default_periode.month),
                                key="input_ewd")
        volume = st.number_input("Volume Aktual", value=0.0, format="%.2f", key="input_volume")

        submit = st.button("Simpan", type="primary")
        if submit:
            if periode in df.index:
                st.toast("Periode sudah ada.", icon="⚠️")
            else:
                df.loc[periode] = {
                    "Tahun": periode.year,
                    "Bulan": periode.month,
                    "Effective Working Days": ewd,
                    "Volume": volume,
                    "BI Rate": bi_rate,
                    "Inflasi": inflasi,
                    "APBN Infra": apbn_infra,
                    "PDB Konstruksi": pdb_konstruksi
                }
                st.session_state.df_vub = df
                update_df_to_gsheet(st.session_state.df_vub)
                st.session_state.reload_data = True
                st.toast("Data berhasil disimpan!", icon="✅")
                time.sleep(1)
                st.rerun()

# --- Edit Data ---
with col2:
    with st.expander("✏️ Edit Data", expanded=True):
        periode_list = df.index.strftime("%Y-%m-%d")
        default_index = len(periode_list) - 1

        periode_edit = st.selectbox("Periode",
                                        periode_list,
                                        index=default_index,
                                        key="edit_periode")

        if periode_edit:
            p = pd.to_datetime(periode_edit)

            def safe_value(val, cast_func, default):
                try:
                    return cast_func(val) if not pd.isna(val) else default
                except:
                    return default

            bi_val = safe_value(df.loc[p, "BI Rate"], float, 0.0)
            inflasi_val = safe_value(df.loc[p, "Inflasi"], float, 0.0)
            apbn_val = safe_value(df.loc[p, "APBN Infra"], float, 0.0)
            pdb_val = safe_value(df.loc[p, "PDB Konstruksi"], float, 0.0)
            ewd_val = safe_value(df.loc[p, "Effective Working Days"], int, 1)
            volume_val = safe_value(df.loc[p, "Volume"], float, 0.0)

            bi_rate = st.number_input("BI Rate", format="%.5f", value=bi_val, key="edit_bi_rate")
            inflasi = st.number_input("Inflasi", format="%.5f", value=inflasi_val, key="edit_inflasi")
            apbn_infra = st.number_input("APBN Infrastruktur", format="%.5f", value=apbn_val, key="edit_apbn_infra")
            pdb_konstruksi = st.number_input("PDB Konstruksi", format="%.5f", value=pdb_val, key="edit_pdb_konstruksi")
            ewd = st.number_input("Hari Kerja Efektif", min_value=1, max_value=31, value=ewd_val, key="edit_ewd")
            volume = st.number_input("Volume Aktual", value=volume_val, key="edit_volume")

            submit_edit = st.button("Perbarui", type="primary")
            if submit_edit:
                df.at[p, "Effective Working Days"] = ewd
                df.at[p, "Volume"] = volume
                df.at[p, "BI Rate"] = bi_rate
                df.at[p, "Inflasi"] = inflasi
                df.at[p, "APBN Infra"] = apbn_infra
                df.at[p, "PDB Konstruksi"] = pdb_konstruksi

                st.session_state.df_vub = df.sort_index()
                update_df_to_gsheet(st.session_state.df_vub)
                st.session_state.reload_data = True
                st.toast("Data berhasil diperbarui!", icon="✅")
                time.sleep(1)
                st.rerun()

# --- Delete Data ---
with col3:
    with st.expander("🗑️ Hapus Data", expanded=True):
        periode_list = df.index.strftime("%Y-%m-%d")
        default_index = len(periode_list) - 1

        periode_hapus = st.selectbox("Periode",
                                        periode_list,
                                        index=default_index,
                                        key="delete_selectbox")

        p = pd.to_datetime(periode_hapus)
        confirm = st.checkbox("Saya yakin ingin menghapus data ini")
        submit_delete = st.button("Hapus", type="primary")

        if submit_delete and confirm:
            df = df.drop(index=p)
            st.session_state.df_vub = df.sort_index()
            update_df_to_gsheet(st.session_state.df_vub)
            st.toast("Data berhasil dihapus!", icon="🗑️")
            time.sleep(1)
            st.rerun()
        elif submit_delete:
            st.toast("Mohon centang konfirmasi terlebih dahulu.", icon="⚠️")

with st.expander("✏️ Update Data Asumsi"):
    updated_assumptions = st.data_editor(
        forecasting_assumptions,
        use_container_width=True
    )

    # cek apakah sudah ada di session_state
    if "df_forecasting_assumptions" not in st.session_state:
        st.session_state.df_forecasting_assumptions = forecasting_assumptions

    # hanya update jika ada perubahan nyata
    if not updated_assumptions.equals(st.session_state.df_forecasting_assumptions):
        # simpan ke session_state
        st.session_state.df_forecasting_assumptions = updated_assumptions

        # update ke Google Sheets
        update_df_to_gsheet(
            updated_assumptions,
            sheet_name="Forecasting VUB"
        )

        st.toast("Data asumsi berhasil diperbarui!", icon="✅")
        time.sleep(1)
        st.rerun()
