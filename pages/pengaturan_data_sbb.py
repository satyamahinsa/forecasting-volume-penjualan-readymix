import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import requests
import html
from bs4 import BeautifulSoup
import holidays
from datetime import date, timedelta
import time

conn = st.connection("gsheets", type=GSheetsConnection)

def reload_df(conn):
    df = conn.read(worksheet="SBB")
    df["Periode"] = pd.to_datetime(df["Periode"])
    df.set_index("Periode", inplace=True)
    return df.sort_index()

def update_df_to_gsheet(df):
    conn.update(worksheet="SBB", data=df.reset_index())

def get_effective_working_days(year, month):
    indo_holidays = holidays.country_holidays('ID', years=[year])
    start_date = date(year, month, 1)
    end_date = pd.Period(f"{year}-{month:02}").end_time.date()
    current = start_date
    workdays = 0
    while current <= end_date:
        if current not in indo_holidays:
            workdays += 1
        current += timedelta(days=1)
    return workdays

def data_scraping():
    df = st.session_state.df_sbb.copy()

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
    df_inflation = df_inflation[(df_inflation.index.year >= 2021)]

    url = f'https://webapi.bps.go.id/v1/api/list/model/data/lang/ind/domain/0000/var/379/key/{API_KEY}?th=2020-2025'
    data = requests.get(url).json().get('datacontent', {})
    bi_data = []
    for kode, val in data.items():
        timecode = kode[6:]
        try:
            tahun = 2000 + int(timecode[:2])
            bulan = int(timecode[2:]) if len(timecode) == 4 else int(timecode[2])
            if 2021 <= tahun <= 2025 and 1 <= bulan <= 12:
                bi_data.append({'Tahun': tahun, 'Bulan': bulan, 'BI Rate': float(val)})
        except:
            continue

    df_bi = pd.DataFrame(bi_data)
    df_bi['Periode'] = pd.to_datetime({'year': df_bi['Tahun'], 'month': df_bi['Bulan'], 'day': 1})
    df_bi.set_index('Periode', inplace=True)
    df_bi.drop(columns=['Tahun', 'Bulan'], inplace=True)
    df_bi['BI Rate'] = df_bi['BI Rate'] / 100

    combined = df.copy()
    all_periodes = set(df_inflation.index).union(df_bi.index)

    for p in sorted(all_periodes):
        inflasi = df_inflation.loc[p, 'Inflasi'] if p in df_inflation.index else None
        bi_rate = df_bi.loc[p, 'BI Rate'] if p in df_bi.index else None
        if p in combined.index:
            if inflasi is not None:
                combined.at[p, 'Inflasi'] = inflasi
            if bi_rate is not None:
                combined.at[p, 'BI Rate'] = bi_rate
        else:
            combined.loc[p] = {
                'Tahun': p.year,
                'Bulan': p.month,
                'BI Rate': bi_rate if not pd.isna(bi_rate) else 0,
                'Inflasi': inflasi if not pd.isna(inflasi) else 0,
                'APBN Infra': 0,
                'PDB Konstruksi': 0,
                'Effective Working Days': get_effective_working_days(p.year, p.month),
                'Volume': 0
            }

    st.session_state.df_sbb = combined.sort_index()
    update_df_to_gsheet(st.session_state.df_sbb)
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
