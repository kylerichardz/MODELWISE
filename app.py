import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import sys

# This must be the first Streamlit command
st.set_page_config(
    page_title="ML Model Selection Advisor",
    page_icon="🤖",
    layout="wide"
)

# Debug info
st.write("Debug Info:")
st.write("- Python version:", sys.version)
st.write("- Streamlit version:", st.__version__)

# Basic app functionality
st.title("🤖 ML Model Selection Advisor")

# Load sample data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Show data
st.dataframe(df.head())