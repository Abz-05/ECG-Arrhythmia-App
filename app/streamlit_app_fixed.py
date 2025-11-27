import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
import os

# ========================================
# SETUP & IMPORTS
# ========================================

# Add src to path if needed (though utils.py handles this, we need to find utils first)
# We'll use a simplified version of the path finder just to import utils
try:
    # Try direct import if running from root
    from src.utils import PROJECT_ROOT, Config, find_model_path
    from src.model import create_model
except ImportError:
    # If running from app/ directory
    current = Path(__file__).resolve().parent
    root = current.parent
    sys.path.append(str(root / 'src'))
    try:
        from utils import PROJECT_ROOT, Config, find_model_path
        from model import create_model
    except ImportError:
        st.error("Critical Error: Could not import source modules. Please run from project root.")
        st.stop()

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="ECG Arrhythmia Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS - ENHANCED DARK MODE DESIGN
# ========================================

st.markdown("""
    <style>
    /* Main header with gradient and glow effect */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
    }
    
    /* Info box with dark glass morphism */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid #667eea;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .info-box h3 {
        color: #667eea;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .info-box p {
        color: #e0e0e0;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .info-box strong {
        color: #a78bfa;
        font-weight: 600;
    }
    
    /* Prediction box with dynamic gradient borders */
    .prediction-box {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.95) 0%, rgba(50, 50, 50, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        padding: 3px;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Metric cards with hover effect */
    .metric-card {
        background: linear-gradient(135deg, rgba(40, 40, 40, 0.8) 0%, rgba(60, 60, 60, 0.8) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Section headers */
    h2, h3 {
        color: #a78bfa !important;
        font-weight: 700;
    }
    
    /* Stats section enhancement */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: scale(1.05);
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .feature-card h3 {
        color: #667eea !important;
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
    }
    
    .feature-card p {
        color: #b8b8b8;
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(30, 30, 30, 0.95) 0%, rgba(20, 20, 20, 0.95) 100%);
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: rgba(40, 40, 40, 0.8) !important;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid;
    }
    
    /* Footer enhancement */
    .footer-enhanced {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 2rem;
    }
    
    .footer-enhanced p {
        color: #a78bfa;
        margin: 0.5rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(30, 30, 30, 0.6);
        border-radius: 8px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .status-active { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    
    /* Risk level badges */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# MODEL LOADING
# ========================================

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    model_path = find_model_path()
    
    if model_path is None:
        return None, None, None
    
    try:
        model = create_model(num_classes=Config.NUM_CLASSES, device=Config.DEVICE)
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint.get('val_acc', None), str(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# ========================================
# PREPROCESSING
# ========================================

def preprocess_signal(signal):
    """Preprocess ECG signal for model input"""
    # Ensure correct length
    if len(signal) != 187:
        signal = np.interp(
            np.linspace(0, len(signal)-1, 187),
            np.arange(len(signal)),
            signal
        )
    
    # Z-score normalization
    mean = signal.mean()
    std = signal.std()
    if std > 1e-6:
        signal = (signal - mean) / std
    
    # Convert to tensor: (1, 1, 187)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return signal_tensor

# ========================================
# PREDICTION
# ========================================

def predict(model, signal_tensor):
    """Make prediction on ECG signal"""
    with torch.no_grad():
        signal_tensor = signal_tensor.to(Config.DEVICE)
        outputs = model(signal_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class] * 100
    
    return predicted_class, probabilities, confidence

# ========================================
# VISUALIZATIONS
# ========================================

def plot_ecg_signal(signal, predicted_class=None):
    """Create interactive ECG plot"""
    fig = go.Figure()
    
    # Main signal
    fig.add_trace(go.Scatter(
        y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Sample: %{x}<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    # Add medical grid
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#ffcccc',
        minor=dict(showgrid=True, gridwidth=0.5, gridcolor='#ffe6e6')
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#ffcccc',
        minor=dict(showgrid=True, gridwidth=0.5, gridcolor='#ffe6e6')
    )
    
    # Title with prediction
    title_text = 'ECG Signal Visualization'
    if predicted_class is not None:
        class_name = Config.CLASS_NAMES[predicted_class]
        color = Config.CLASS_COLORS[predicted_class]
        title_text += f' - <span style="color:{color}">Predicted: {class_name}</span>'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Sample Index (≈1.5 seconds @ 125 Hz)',
        yaxis_title='Normalized Amplitude (mV)',
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    return fig

def plot_probability_bar(probabilities):
    """Create bar chart of class probabilities"""
    fig = go.Figure()
    
    colors = [Config.CLASS_COLORS[i] for i in range(len(probabilities))]
    
    fig.add_trace(go.Bar(
        x=Config.CLASS_NAMES,
        y=probabilities * 100,
        marker=dict(color=colors),
        text=[f'{p*100:.1f}%' for p in probabilities],
        textposition='outside',
        hovertemplate='%{x}<br>Probability: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prediction Probabilities by Class',
        xaxis_title='Arrhythmia Class',
        yaxis_title='Probability (%)',
        yaxis_range=[0, 105],
        template='plotly_white',
        height=400
    )
    
    return fig

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ ECG Arrhythmia Classification System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
        <h3>🏥 AI-Powered Clinical Decision Support</h3>
        <p>
        This system uses a hybrid CNN-LSTM deep learning architecture to classify cardiac arrhythmias 
        from ECG signals. Trained on the MIT-BIH Arrhythmia Database with 109,000+ annotated heartbeats.
        </p>
        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for <b>educational and research purposes only</b>. 
        It is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals 
        for medical decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ System Settings")
    st.sidebar.write(f"**Project Root:** `{PROJECT_ROOT.name}/`")
    st.sidebar.write(f"**Device:** {Config.DEVICE.upper()}")
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, val_acc, model_path = load_model()
    
    if model is None:
        st.error("❌ **Model Not Found**")
        st.warning("Please ensure you have trained the model first:")
        st.code("python src/train.py", language="bash")
        st.info(f"Searched in: {PROJECT_ROOT / 'models'}")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"📂 **Model:** `{Path(model_path).name}`")
    if val_acc:
        st.sidebar.metric("Validation Accuracy", f"{val_acc:.2f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Input Data")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload ECG CSV File",
        type=['csv'],
        help="CSV file with 187 ECG sample values"
    )
    
    # Sample data
    use_sample = st.sidebar.checkbox("Use Sample Data", value=False)
    
    signal = None
    
    if use_sample:
        st.sidebar.info("📊 Using synthetic demo data")
        sample_type = st.sidebar.selectbox(
            "Select Pattern:",
            ['Normal', 'Ventricular', 'Supraventricular']
        )
        
        # Generate synthetic patterns
        if sample_type == 'Normal':
            signal = np.sin(np.linspace(0, 4*np.pi, 187)) * 0.8
            signal += np.sin(np.linspace(0, 20*np.pi, 187)) * 0.3
        elif sample_type == 'Ventricular':
            signal = np.sin(np.linspace(0, 3*np.pi, 187)) * 1.2
            signal[80:110] += 1.5  # Wide QRS
        else:
            signal = np.sin(np.linspace(0, 5*np.pi, 187)) * 0.7
            signal += np.sin(np.linspace(0, 25*np.pi, 187)) * 0.2
        
        signal += np.random.randn(187) * 0.05
        
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            
            if len(df) > 1:
                row_idx = st.sidebar.selectbox(
                    "Select Row:",
                    range(min(10, len(df))),
                    format_func=lambda x: f"Beat #{x+1}"
                )
            else:
                row_idx = 0
            
            signal = df.iloc[row_idx, :187].values.astype(np.float32)
            st.sidebar.success(f"✅ Loaded {len(signal)} samples")
            
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # ========================================
    # MAIN CONTENT
    # ========================================
    
    if signal is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 ECG Signal")
            fig_signal = plot_ecg_signal(signal)
            st.plotly_chart(fig_signal, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Signal Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Length', 'Mean', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{len(signal)} samples",
                    f"{signal.mean():.4f}",
                    f"{signal.std():.4f}",
                    f"{signal.min():.4f}",
                    f"{signal.max():.4f}",
                    f"{signal.max() - signal.min():.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🚀 Classify Arrhythmia", type="primary"):
            with st.spinner('🔬 Analyzing ECG signal...'):
                signal_tensor = preprocess_signal(signal)
                predicted_class, probabilities, confidence = predict(model, signal_tensor)
                
                # Results
                st.markdown("## 📋 Classification Results")
                
                class_name = Config.CLASS_NAMES[predicted_class]
                class_color = Config.CLASS_COLORS[predicted_class]
                risk_level = Config.CLASS_RISKS[predicted_class]
                
                # Main prediction box
                st.markdown(f"""
                    <div class="prediction-box" style="border-left: 5px solid {class_color};">
                        <h2 style="color: {class_color}; margin: 0;">
                            {class_name}
                        </h2>
                        <h3 style="margin-top: 1rem;">
                            Confidence: {confidence:.1f}%
                        </h3>
                        <p style="margin-top: 0.5rem; color: #666;">
                            Risk Level: <b>{risk_level}</b>
                        </p>
                        <p style="margin-top: 1rem; color: #555; font-size: 1.1rem;">
                            {Config.CLASS_DESCRIPTIONS[class_name]}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.subheader("📊 Probability Distribution")
                fig_probs = plot_probability_bar(probabilities)
                st.plotly_chart(fig_probs, use_container_width=True)
                
                # Detailed table
                with st.expander("📈 View Detailed Probabilities"):
                    probs_df = pd.DataFrame({
                        'Class': Config.CLASS_NAMES,
                        'Probability': [f"{p*100:.2f}%" for p in probabilities],
                        'Risk Level': [Config.CLASS_RISKS[i] for i in range(5)],
                        'Description': [Config.CLASS_DESCRIPTIONS[name] for name in Config.CLASS_NAMES]
                    })
                    st.dataframe(probs_df, use_container_width=True, hide_index=True)
                
                # Clinical interpretation
                st.markdown("---")
                st.subheader("💡 Clinical Interpretation")
                
                if predicted_class == 0:
                    st.success("✅ **Normal rhythm** - No immediate concerns. Continue routine monitoring.")
                elif predicted_class == 1:
                    st.warning("⚠️ **Supraventricular ectopy** - Usually benign. Consider clinical correlation and symptom assessment.")
                elif predicted_class == 2:
                    st.error("🚨 **Ventricular ectopy** - May require further evaluation. Consider frequency and clinical context.")
                elif predicted_class == 3:
                    st.info("ℹ️ **Fusion beat** - Represents hybrid ventricular activation. Document and monitor.")
                else:
                    st.info("❓ **Unclassifiable** - May be paced rhythm or artifact. Expert review recommended.")
                
    else:
        # Instructions
        st.info("👆 **Get Started:** Upload an ECG CSV file or select sample data from the sidebar")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📤 Upload Data")
            st.write("Upload a CSV file with 187 ECG samples per beat")
        
        with col2:
            st.markdown("### 📊 Use Samples")
            st.write("Try synthetic demo data to test the system")
        
        with col3:
            st.markdown("### 🔬 Get Results")
            st.write("View predictions with confidence scores")
        
        with st.expander("📄 CSV File Format"):
            st.markdown("""
                **Requirements:**
                - **187 columns** (ECG sample values)
                - **Numeric values** (float)
                - **No header** row
                - Each row = one heartbeat
                
                **Example:**
                ```
                -0.123,0.456,0.789,...(187 values total)
                0.234,-0.567,0.891,...(187 values total)
                ```
                
                **Download Dataset:**
                - [Kaggle MIT-BIH](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
                - [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1.5rem 0;">
            <p><b>ECG Arrhythmia Classification System v1.0</b></p>
            <p>CNN-LSTM Architecture | MIT-BIH Database | PyTorch Framework</p>
            <p style="font-size: 0.9rem; color: #999;">
                🔬 For Research & Educational Use Only | Not for Clinical Diagnosis
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()