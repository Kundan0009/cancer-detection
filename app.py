import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Cancer Classification using GA-PSO & Transformer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Cancer Classification System")
st.write("Using Hybrid GA-PSO Optimization and Transformer Networks")

# Check if modules are available
try:
    from data_preprocessing import DataPreprocessor, generate_synthetic_dataset
    from optimization import HybridGAPSO
    from transformer_model import GeneTransformerClassifier
    from evaluation import ModelEvaluator
    from interpretability import GeneImportanceAnalyzer
    modules_available = True
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.info("Running in demo mode with basic functionality only.")
    modules_available = False

if modules_available:
    # Full app functionality
    st.success("✅ All modules loaded successfully!")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Data generation demo
    if st.sidebar.button("Generate Demo Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                dataset_path = generate_synthetic_dataset(
                    n_samples=500,
                    n_genes=1000,
                    n_classes=3,
                    n_informative=50,
                    output_file="demo_data.csv"
                )
                
                preprocessor = DataPreprocessor(random_state=42)
                data = preprocessor.preprocess_pipeline(dataset_path, test_size=0.2)
                
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success("✅ Demo data generated!")
            except Exception as e:
                st.error(f"Error generating data: {e}")
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Genes", data['n_features'])
        col2.metric("Training Samples", len(data['X_train']))
        col3.metric("Test Samples", len(data['X_test']))
        col4.metric("Classes", data['n_classes'])
        
        # Show class distribution
        train_dist = pd.Series(data['y_train']).value_counts().sort_index()
        st.bar_chart(train_dist)
        
else:
    # Demo mode
    st.info("📊 Demo Mode - Basic Functionality")
    
    if st.button("Generate Sample Data"):
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_genes = 5000
        
        data = pd.DataFrame({
            f'Gene_{i}': np.random.randn(n_samples) 
            for i in range(min(10, n_genes))  # Show only first 10 genes
        })
        data['Cancer_Type'] = np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples)
        
        st.success("✅ Sample data generated!")
        st.dataframe(data.head())
        
        # Basic visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        data['Cancer_Type'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Cancer Type Distribution')
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("1. Click 'Generate Demo Data' to create synthetic gene expression data")
st.markdown("2. The system will show data overview and class distribution")
st.markdown("3. Full ML pipeline available when all modules are loaded")