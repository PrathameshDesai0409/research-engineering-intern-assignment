import streamlit as st
import pandas as pd

# Set the page title
st.set_page_config(page_title="Social Media Analytics Dashboard", layout="wide")

# Title of the dashboard
st.title("📊 Social Media Analytics Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.write(df.head())

st.write("👈 Upload a dataset to start exploring!")
