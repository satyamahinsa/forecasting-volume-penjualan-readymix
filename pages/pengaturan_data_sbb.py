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

def reload_df(conn):
    df = conn.read(worksheet="SBB")
    df["Periode"] = pd.to_datetime(df["Periode"])
    df.set_index("Periode", inplace=True)
    return df.sort_index()

def update_df_to_gsheet(df, sheet_name="SBB"):
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

def data_scraping():
    df = st.session_state.df_sbb.copy()

    # data scraping & forecasting inflasi
    API_KEY = '01885d016e24d4a4bce1862bdd1c6ad7'
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
    train_inflation = df_inflation[(df_inflation.index.year >= 2006) & (df_inflation.index.year < 2025)]
    test_inflation = df_inflation[(df_inflation.index.year == 2025)]
    model_inflation = pm.auto_arima(
        train_inflation,
        seasonal=True,
        m=12,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    order = model_inflation.get_params()['order']
    seasonal_order = model_inflation.get_params()['seasonal_order']
    sarimax_model_inflation = SARIMAX(train_inflation, order=order, seasonal_order=seasonal_order)
    sarimax_model_inflation_fit = sarimax_model_inflation.fit()
    df_inflation = df_inflation[(df_inflation.index.year >= 2020)]
    forecast_inflation_future = sarimax_model_inflation_fit.forecast(steps=12)
    
    df_forecast_inflation_future = pd.DataFrame({
        'Inflasi': forecast_inflation_future.values
    }, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))

    # data scraping and forecasting BI Rate
    API_KEY = '01885d016e24d4a4bce1862bdd1c6ad7'
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
    train_bi_rate = df_bi_rate[(df_bi_rate.index.year >= 2009) & (df_bi_rate.index.year < 2025)]
    test_bi_rate = df_bi_rate[(df_bi_rate.index.year == 2025)]
    model_bi_rate = pm.auto_arima(
        train_bi_rate,
        seasonal=True,
        m=12,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    order = model_bi_rate.get_params()['order']
    seasonal_order = model_bi_rate.get_params()['seasonal_order']
    sarimax_model_bi_rate = SARIMAX(train_bi_rate, order=order, seasonal_order=seasonal_order)
    sarimax_model_bi_rate_fit = sarimax_model_bi_rate.fit()
    df_bi_rate = df_bi_rate[(df_bi_rate.index.year >= 2020)]
    forecast_bi_rate_future = sarimax_model_bi_rate_fit.forecast(steps=12)
    
    df_forecast_bi_rate = pd.DataFrame({
        'BI Rate': forecast_bi_rate_future.values
    }, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))
    st.dataframe(df_forecast_bi_rate, use_container_width=True)
    
    # data scraping dan forecasting apbn infra 
    url = "https://media.kemenkeu.go.id/SinglePage/custompage?p=/Pages/Home/Anggaran-Infrastruktur"
    data = json.loads(requests.get(url).text)
    data_anggaran_infra = data['Data']['Content']

    data_tahun = []
    data_jumlah_infrastruktur = []

    for item in data_anggaran_infra:
        data_tahun.append(item['Tahun'])
        data_jumlah_infrastruktur.append(item['Jumlah'])

    df_apbn = pd.DataFrame({'Tahun': data_tahun, 'APBN Infrastruktur': data_jumlah_infrastruktur})
    df_apbn['Tahun'] = df_apbn['Tahun'].astype(int)
    df_apbn['APBN Infrastruktur'] = df_apbn['APBN Infrastruktur'].str.replace(',', '.').astype(float)
    url_apbn_2025 = 'https://ekonomi.bisnis.com/read/20240816/45/1791651/anggaran-infrastruktur-rp400-triliun-untuk-proyek-prioritas-di-2025-apa-saja'

    results = requests.get(url_apbn_2025)
    soup = BeautifulSoup(results.text, 'html.parser')
    first_p_element = soup.find('article').find('p')
    text = first_p_element.get_text(strip=True)
    year_pattern = r'infrastruktur(\d{4})'
    amount_pattern = r'Rp(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)\s*triliun'

    # Mencari tahun pertama yang ditemukan
    tahun_2025 = re.search(year_pattern, text)
    tahun_2025 = tahun_2025.group(1) if tahun_2025 else None

    # Mencari angka desimal pertama yang ditemukan
    infrastruktur_2025 = re.search(amount_pattern, text)
    infrastruktur_2025 = infrastruktur_2025.group(1) if infrastruktur_2025 else None

    # Menghapus titik dari angka dan mengganti koma dengan titik untuk konversi ke float
    if infrastruktur_2025:
        infrastruktur_2025 = infrastruktur_2025.replace('.', '').replace(',', '.')
        infrastruktur_2025 = float(infrastruktur_2025)

    data_2025 = pd.DataFrame({'Tahun': [tahun_2025], 'APBN Infrastruktur': [infrastruktur_2025]})
    data_2025['Tahun'] = data_2025['Tahun'].astype(int)
    data_2025['APBN Infrastruktur'] = data_2025['APBN Infrastruktur'].astype(float)
    df_apbn = pd.concat([df_apbn, data_2025], ignore_index=True)
    result = []  # untuk menyimpan baris result
    last_year = df.index.max().year
    last_month = df.index.max().month

    # Proses data yang sudah ada
    for index, row in df.iterrows():
        year = row.name.year
        month = row.name.month
        
        apbn_value = df_apbn[df_apbn['Tahun'] == year]['APBN Infrastruktur'].values[0]
        comparison_year = 2024 if year == 2025 else year
        
        volume = row['Volume']
        total_volume = df[df.index.year == comparison_year]['Volume'].sum()
        ratio = volume / total_volume if total_volume > 0 else 0
        apbn_per_month = apbn_value * ratio
        
        result.append({
            'Tahun': year,
            'Bulan': month,
            'Volume': volume,
            'Rasio': ratio,
            'APBN Infrastruktur': apbn_per_month,
            'Sumber': 'Aktual'
        })
        
    for month in range(last_month+1, 13):
        if not ((df.index.year == last_year) & (df.index.month == month)).any():
            # pakai tahun sebelumnya
            if ((df.index.year == last_year-1) & (df.index.month == month)).any():
                prev_row = df[(df.index.year == last_year-1) & (df.index.month == month)].iloc[0]
                prev_volume = prev_row['Volume']
            else:
                prev_volume = 0
            
            total_volume_prev = df[df.index.year == last_year-1]['Volume'].sum()
            ratio = prev_volume / total_volume_prev if total_volume_prev > 0 else 0
            apbn_per_month = apbn_value * ratio
            
            result.append({
                'Tahun': last_year,
                'Bulan': month,
                'Volume': prev_volume,
                'Rasio': ratio,
                'APBN Infrastruktur': apbn_per_month,
                'Sumber': 'Prediksi'
            })

    # Buat DataFrame result
    df_apbn_infra = pd.DataFrame(result).sort_values(by=['Tahun', 'Bulan']).reset_index(drop=True)
    df_apbn_infra['Periode'] = pd.to_datetime(
        df_apbn_infra[['Tahun', 'Bulan']].rename(columns={'Tahun': 'year', 'Bulan': 'month'}).assign(day=1)
    )
    df_apbn_infra.set_index('Periode', inplace=True)
    forecast_apbn_infra_future = df_apbn_infra['APBN Infrastruktur'][-12:]
    
    df_forecast_apbn_infra = pd.DataFrame({
        'APBN Infrastruktur': forecast_apbn_infra_future.values
    }, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))
    
    # data scraping and forecasting pdb konstruksi
    df_pdb_konstruksi = df['PDB Konstruksi']
    train_pdb_konstruksi = df_pdb_konstruksi[(df_pdb_konstruksi.index.year >= 2020) & (df_pdb_konstruksi.index.year < 2025)]
    test_pdb_konstruksi = df_pdb_konstruksi[(df_pdb_konstruksi.index.year == 2025)]
    model_pdb_konstruksi = pm.auto_arima(
        train_pdb_konstruksi,
        seasonal=True,
        m=12,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    order = model_pdb_konstruksi.get_params()['order']
    seasonal_order = model_pdb_konstruksi.get_params()['seasonal_order']
    sarimax_model_pdb_konstruksi = SARIMAX(train_pdb_konstruksi, order=order, seasonal_order=seasonal_order)
    sarimax_model_pdb_konstruksi_fit = sarimax_model_pdb_konstruksi.fit()
    forecast_pdb_konstruksi_future = sarimax_model_pdb_konstruksi_fit.forecast(steps=12)

    df_forecast_pdb_konstruksi = pd.DataFrame({
        'PDB Konstruksi': forecast_pdb_konstruksi_future.values
    }, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))

    combined = st.session_state.df_sbb.copy()
    all_periodes = set(df_inflation.index) | set(df_bi_rate.index) | set(df_apbn_infra.index)

    for p in sorted(all_periodes):
        inflasi = df_inflation.loc[p, 'Inflasi'] if p in df_inflation.index else 0
        bi_rate = df_bi_rate.loc[p, 'BI Rate'] if p in df_bi_rate.index else 0
        apbn_infra = df_apbn_infra.loc[p, 'APBN Infrastruktur'] if p in df_apbn_infra.index else 0
        pdb_konstruksi = df_forecast_pdb_konstruksi.loc[p, 'PDB Konstruksi'] if p in df_forecast_pdb_konstruksi.index else 0
        
        if p in combined.index:
            # jika sudah ada, update nilainya
            if inflasi != 0:
                combined.at[p, 'Inflasi'] = inflasi
            else:
                combined.at[p, 'Inflasi'] = df_forecast_inflation_future.loc[p, 'Inflasi']
            if bi_rate != 0:
                combined.at[p, 'BI Rate'] = bi_rate
            else:
                combined.at[p, 'BI Rate'] = df_forecast_bi_rate.loc[p, 'BI Rate']
            if apbn_infra != 0 or combined.at[p, 'Volume'] > 0:
                combined.at[p, 'APBN Infra'] = apbn_infra
            elif (apbn_infra is None) or (pd.isna(apbn_infra)) or (apbn_infra == 0):
                tahun_lalu = p.year - 1
                periode_lalu = pd.Timestamp(year=tahun_lalu, month=p.month, day=1)
                if periode_lalu in df_apbn_infra.index:
                    apbn_infra = df_apbn_infra.loc[periode_lalu, 'APBN Infrastruktur']
                    combined.at[p, 'APBN Infra'] = apbn_infra
            if combined.at[p, 'Effective Working Days'] != 0:
                combined.at[p, 'Effective Working Days'] = get_effective_working_days(p.year, p.month)
        else:
            # jika belum ada, buat baris baru
            if (apbn_infra is None) or (pd.isna(apbn_infra)) or (apbn_infra == 0):
                tahun_lalu = p.year - 1
                periode_lalu = pd.Timestamp(year=tahun_lalu, month=p.month, day=1)
                if periode_lalu in df_apbn_infra.index:
                    apbn_infra = df_apbn_infra.loc[periode_lalu, 'APBN Infrastruktur']
                    
            combined.loc[p] = {
                'Tahun': p.year,
                'Bulan': p.month,
                'BI Rate': df_forecast_bi_rate.loc[p, 'BI Rate'],
                'Inflasi': df_forecast_inflation_future.loc[p, 'Inflasi'],
                'APBN Infra': apbn_infra,
                'PDB Konstruksi': pdb_konstruksi,
                'Effective Working Days': get_effective_working_days(p.year, p.month),
                'Volume': 0
            }
    
    st.session_state.df_sbb = combined.sort_index()
    update_df_to_gsheet(st.session_state.df_sbb, sheet_name="SBB")
    st.session_state.reload_data = True


st.title("⚙️ Pengaturan Data SBB")

if "df_sbb" not in st.session_state:
    st.session_state.df_sbb = reload_df(conn)

df = st.session_state.df_sbb
st.dataframe(df, use_container_width=True)

# --- Update Data Otomatis ---
with st.expander("🔄 Update Data Otomatis"):
    if st.button("Ambil Data dari API", type="primary"):
        with st.spinner("Mengambil dan memproses data..."):
            try:
                data_scraping()
                st.toast("Data berhasil diperbarui!", icon="✅")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.toast(f"Gagal mengambil data: {e}", icon="❌")

col1, col2, col3 = st.columns(3)

# --- Input Data Baru ---
with col1:
    with st.expander("➕ Input Data Baru"):
        last_periode = df.index.max() if not df.empty else pd.Timestamp.today()
        default_periode = (last_periode + pd.offsets.MonthBegin(1)).replace(day=1)

        periode = st.date_input("Periode", value=default_periode, format="YYYY-MM-DD", key="input_periode")
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
                    "Effective Working Days": ewd,
                    "Volume": volume,
                    "BI Rate": bi_rate,
                    "Inflasi": inflasi,
                    "APBN Infra": apbn_infra,
                    "PDB Konstruksi": pdb_konstruksi
                }
                st.session_state.df_sbb = df.sort_index()
                update_df_to_gsheet(st.session_state.df_sbb)
                st.session_state.reload_data = True
                st.toast("Data berhasil disimpan!", icon="✅")
                time.sleep(1)
                st.rerun()

# --- Edit Data ---
with col2:
    with st.expander("✏️ Edit Data"):
        last_periode = df.index.max() if not df.empty else pd.Timestamp.today()
        periode_edit = st.date_input("Periode", value=last_periode, format="YYYY-MM-DD", key="edit_periode")

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

                st.session_state.df_sbb = df.sort_index()
                update_df_to_gsheet(st.session_state.df_sbb)
                st.session_state.reload_data = True
                st.toast("Data berhasil diperbarui!", icon="✅")
                time.sleep(1)
                st.rerun()

# --- Delete Data ---
with col3:
    with st.expander("🗑️ Hapus Data"):
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
            st.session_state.df_sbb = df.sort_index()
            update_df_to_gsheet(st.session_state.df_sbb)
            st.toast("Data berhasil dihapus!", icon="🗑️")
            time.sleep(1)
            st.rerun()
        elif submit_delete:
            st.toast("Mohon centang konfirmasi terlebih dahulu.", icon="⚠️")
