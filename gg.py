import os
import streamlit as st

if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.warning("Hugging Face token not found in secrets. Please add it via secrets.toml.")
