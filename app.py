import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
import google.generativeai as genai
import os

def get_current_context():
    context = "Current App State:\n"
    
    if 'df' in locals():
        context += f"""
Dataset: {dataset_option if 'dataset_option' in locals() else 'Not selected'}
Number of samples: {df.shape[0]}
Number of features: {df.shape[1]}
Features: {', '.join(df.columns.tolist())}
"""
        if 'target_column' in locals():
            context += f"Target variable: {target_column}\n"
            
        if 'n_samples' in locals():
            context += "\nRecommended Models:\n"
            if n_samples < 1000:
                context += "- Linear/Logistic Regression (small dataset)\n"
                context += "- Decision Trees\n"
            elif n_samples < 10000:
                context += "- Random Forest (medium dataset)\n"
                context += "- Support Vector Machines\n"
            else:
                context += "- Gradient Boosting (XGBoost, LightGBM)\n"
                context += "- Neural Networks\n"
    
    return context

# This must be the first Streamlit command
st.set_page_config(
    page_title="ML Model Selection Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1f77b4;
    }
    .stRadio>label {
        font-weight: bold;
        color: #2c3e50;
    }
    .stMarkdown {
        font-family: 'Helvetica Neue', sans-serif;
    }
    div[data-testid="stHeader"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stChatMessage {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    div.stChatMessage p,
    div.stChatMessage span,
    div.stChatMessage text,
    div.stChatMessage div,
    .stChatMessage * {
        color: #000000 !important;
        font-size: 16px !important;
    }
    div.stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    div.stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }
    .stChatInputContainer textarea,
    .stChatInputContainer * {
        color: #000000 !important;
    }
    [data-testid="stChatMessageContent"] * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; padding: 1.5rem;'>
        🤖 Machine Learning Model Selection Advisor
    </h1>
""", unsafe_allow_html=True)

# Update the Google AI configuration section with better error handling
st.markdown("### 🔄 App Status")
try:
    if 'GOOGLE_API_KEY' in st.secrets:
        st.success("✅ API Key found")
        genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
        model = genai.GenerativeModel('gemini-pro')
        st.success("✅ Google AI Model configured successfully")
    else:
        st.error("❌ API Key not found in secrets")
        st.info("Please configure the GOOGLE_API_KEY in Streamlit Cloud settings")
        st.stop()
except Exception as e:
    st.error(f"❌ Error configuring Google AI: {str(e)}")
    st.info("Check your API key and internet connection")
    st.stop()

# Create two columns for main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Choose a Dataset")
    dataset_option = st.radio(
        label="Dataset Selection",
        options=["Upload My Own", "Iris (Small Classification)", "Breast Cancer (Medium Classification)", 
                "Synthetic (Large Classification)", "Synthetic Regression (Large)"],
        label_visibility="collapsed"
    )

    if dataset_option == "Upload My Own":
        uploaded_file = st.file_uploader(
            label="Upload CSV File",
            type="csv",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        # Load sample datasets with enhanced descriptions
        if dataset_option == "Iris (Small Classification)":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("🌸 Iris Dataset: Perfect for beginners! Classify iris plants into three species based on flower measurements.")
        
        elif dataset_option == "Breast Cancer (Medium Classification)":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("🏥 Breast Cancer Dataset: Real-world medical data for binary classification.")
        
        elif dataset_option == "Synthetic (Large Classification)":
            X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, 
                                    n_redundant=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            df['target'] = y
            st.success("🤖 Synthetic Classification: Large-scale dataset with known patterns.")
        
        else:  # Synthetic Regression
            X = np.random.randn(10000, 10)
            y = np.sum(X[:, :3], axis=1) + np.random.randn(10000) * 0.1
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            df['target'] = y
            st.success("📈 Synthetic Regression: Complex numerical predictions with clear patterns.")

with col2:
    st.markdown("### 🎯 Quick Stats")
    if 'df' in locals():
        st.metric("Samples", f"{df.shape[0]:,}")
        st.metric("Features", df.shape[1] - 1)
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

# Main content area
if 'df' in locals():
    st.markdown("---")
    
    # Dataset preview in an expander
    with st.expander("📊 Dataset Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True)
    
    # Target selection with better styling
    st.markdown("### 🎯 Select Target Variable")
    target_column = st.selectbox(
        label="Target Variable",
        options=df.columns.tolist(),
        index=len(df.columns)-1,
        label_visibility="collapsed"
    )
    
    # Analysis button with custom styling
    if st.button("🔍 Analyze Dataset", key="analyze"):
        st.markdown("---")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{len(df):,} samples")
        with col2:
            st.metric("Features", f"{len(df.columns) - 1} columns")
        with col3:
            st.metric("Data Type", "Classification" if len(df[target_column].unique()) < 10 else "Regression")
        
        # Model recommendations in a nice card
        st.markdown("""
            <div style='padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0;'>
                <h3 style='color: #2c3e50;'>🎯 Recommended Models</h3>
        """, unsafe_allow_html=True)
        
        n_samples = len(df)
        if n_samples < 1000:
            st.info("📊 For small datasets (<1000 samples):")
            st.markdown("- ✨ Linear/Logistic Regression (Simple & Interpretable)")
            st.markdown("- 🌳 Decision Trees (Easy to Understand)")
        elif n_samples < 10000:
            st.info("📊 For medium datasets:")
            st.markdown("- 🌲 Random Forest (Robust & Accurate)")
            st.markdown("- 🎯 Support Vector Machines (Powerful for Complex Patterns)")
        else:
            st.info("📊 For large datasets:")
            st.markdown("- 🚀 Gradient Boosting (XGBoost, LightGBM)")
            st.markdown("- 🧠 Neural Networks (Deep Learning)")

# AI Assistant section with better styling
st.markdown("---")
st.markdown("### 🤖 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the models or your data!"):
    context = get_current_context()
    full_prompt = f"""
Context about the current state of the ML Model Selection Advisor:
{context}

User Question: {prompt}

Please provide advice based on this context and your knowledge of machine learning.
"""
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = model.generate_content(full_prompt)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hi! I'm your ML advisor. How can I help you?"})