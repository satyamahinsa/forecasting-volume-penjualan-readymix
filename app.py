import streamlit as st

st.set_page_config(
    page_title="Peramalan Volume Penjualan ReadyMix",
    page_icon="📊",
    layout="wide"
)

pages = st.navigation({
    "Main Menu": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
    ],
    "Tipe": [
        st.Page("pages/sbb.py", title="SBB", icon="📊"),
        st.Page("pages/vub.py", title="VUB", icon="📊"),
    ],
    "Pengaturan": [
        st.Page("pages/pengaturan_data_sbb.py", title="Data SBB", icon="📄"),
        st.Page("pages/pengaturan_data_vub.py", title="Data VUB", icon="📄")
    ]
})

pages.run()