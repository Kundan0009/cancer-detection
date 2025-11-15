import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Cancer Classification Test",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Cancer Classification System")
st.write("Testing basic functionality...")

# Test basic functionality
if st.button("Test"):
    st.success("App is working!")
    
    # Test data generation
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'label': np.random.randint(0, 2, 100)
    })
    
    st.write("Sample data:")
    st.dataframe(data.head())
    
st.info("If you see this, the basic app is running correctly.")