import streamlit as st

# Konfigurasi halaman


st.title("📊 Peramalan Volume Penjualan ReadyMix")

st.write(
    "Aplikasi ini digunakan untuk memprediksi volume penjualan ReadyMix SBB dan VUB selama 1 tahun ke depan berdasarkan data internal perusahaan dan faktor makro eksternal"
)

st.markdown("---")

# Tampilan dua pilihan
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔹 ReadyMix SBB")
    st.write("Lihat tren dan peramalan penjualan ReadyMix SBB selama 1 tahun ke depan.")
    if st.button("📊 Buka Dashboard SBB", type="primary"):
        st.switch_page("pages/sbb.py")

with col2:
    st.subheader("🔸 ReadyMix VUB")
    st.write("Lihat tren dan peramalan penjualan ReadyMix VUB selama 1 tahun ke depan.")
    if st.button("📊 Buka Dashboard VUB", type="primary"):
        st.switch_page("pages/vub.py")

