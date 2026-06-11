import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Try to import joblib, if fail, use pickle as fallback
try:
    import joblib
    print("Using joblib")
except ImportError:
    import pickle as joblib
    print("Using pickle as fallback")

st.set_page_config(page_title="XGBoost M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings using **XGBoost** Regressor*")
st.info("📌 **Model:** XGBoost | **Data:** Synthetic | **Features:** Temperature, Humidity, Hour, Occupants, Floor Area, Retrofit")

# ==========================================
# LOAD XGBOOST MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Try multiple possible paths
    possible_paths = [
        'models/xgboost_model.pkl',
        'xgboost_model.pkl',
        'model.pkl'
    ]
    
    model = None
    model_path_used = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_path_used = path
                break
            except:
                continue
    
    if model is None:
        st.error("❌ XGBoost model not found!")
        st.info("""
        **Troubleshooting:**
        1. Make sure 'xgboost_model.pkl' is in the 'models/' folder
        2. Or upload the model file directly to the root folder
        3. Check the file exists in GitHub repository
        """)
        
        # List files for debugging
        st.write("Files in current directory:")
        for f in os.listdir('.'):
            st.write(f"  - {f}")
        
        if os.path.exists('models'):
            st.write("Files in models/ folder:")
            for f in os.listdir('models'):
                st.write(f"  - {f}")
        
        return None, None
    
    # Load features
    features_paths = [
        'models/xgboost_features.txt',
        'xgboost_features.txt',
        'features.txt'
    ]
    
    features = None
    for path in features_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            break
    
    if features is None:
        # Default features
        features = [
            'temperature', 'humidity', 'hour', 'dayofweek', 'month',
            'floor_area', 'occupants', 'retrofit',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_weekend', 'temp_humidity', 'occ_per_area'
        ]
    
    st.success(f"✅ XGBoost model loaded from {model_path_used}")
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

# ==========================================
# LOAD METRICS (if available)
# ==========================================
@st.cache_resource
def load_metrics():
    metrics_paths = [
        'models/xgboost_metrics.json',
        'xgboost_metrics.json',
        'metrics.json'
    ]
    
    for path in metrics_paths:
        if os.path.exists(path):
            try:
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                continue
    return None

metrics = load_metrics()

if metrics:
    st.sidebar.success("✅ XGBoost model ready!")
else:
    st.sidebar.info("📊 XGBoost model loaded (metrics file not found)")

# ==========================================
# SCALING FACTOR (Malaysia residential)
# ==========================================
SCALING_FACTOR = 10

def scale_prediction(prediction):
    return prediction / SCALING_FACTOR

def convert_energy_unit(prediction_kwh, target_unit):
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Model performance
st.sidebar.markdown("---")
st.sidebar.subheader("📊 XGBoost Performance")

if metrics:
    with st.sidebar.expander("Performance Metrics", expanded=True):
        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
        st.metric("MAE", f"{metrics.get('mae', 0):.2f} kWh")
        if 'rmse' in metrics:
            st.caption(f"RMSE: {metrics['rmse']:.2f} kWh")
else:
    with st.sidebar.expander("Performance Metrics", expanded=True):
        st.info("Metrics file not found")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Settings")

unit_option = st.sidebar.selectbox(
    "Energy Unit Display",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"]
)

st.sidebar.markdown("---")

# Input parameters (Malaysia range)
temp = st.sidebar.slider("🌡️ Temperature (°C)", 22, 35, 28)
humidity = st.sidebar.slider("💧 Humidity (%)", 60, 95, 80)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 60, 300, 120)
occupants = st.sidebar.number_input("👥 Occupants", 1, 8, 4)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# ==========================================
# MAIN CONTENT
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy (XGBoost)", type="primary", use_container_width=True):
        # Predict
        raw_pred = model.predict(features_df)[0]
        pred = scale_prediction(raw_pred)
        
        converted, unit = convert_energy_unit(pred, unit_option)
        
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
        
        if retrofit == 1:
            # Baseline (without retrofit)
            base_df = features_df.copy()
            base_df['retrofit'] = 0
            raw_base = model.predict(base_df)[0]
            base_pred = scale_prediction(raw_base)
            
            savings = base_pred - pred
            savings_pct = (savings / base_pred) * 100
            savings_conv, _ = convert_energy_unit(savings, unit_option)
            
            m2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
            
            st.success(f"💡 XGBoost says: Retrofit saves {savings_conv:.2f} {unit} ({savings_pct:.1f}%)")
            
            # Monthly bill savings
            tariff = 0.52
            monthly_savings = savings * 24 * 30
            monthly_rm = monthly_savings * tariff
            st.info(f"💰 **Estimated monthly bill savings:** RM {monthly_rm:.2f}/month")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8,5))
            bars = ax.bar(['Baseline\n(No Retrofit)', 'Retrofitted'], 
                         [base_pred, pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
            
            for bar, val in zip(bars, [base_pred, pred]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f} kWh', ha='center', fontweight='bold')
            
            ax.set_ylabel('Energy Consumption (kWh)')
            ax.set_title('XGBoost: Retrofit Impact', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Gauge
            fig2, ax2 = plt.subplots(figsize=(8,2.5))
            color = '#2ecc71' if savings_pct > 20 else '#f39c12' if savings_pct > 10 else '#e74c3c'
            label = 'High' if savings_pct > 20 else 'Medium' if savings_pct > 10 else 'Low'
            
            ax2.barh([0], [min(savings_pct,100)], color=color, height=0.4, edgecolor='black')
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0,100)
            ax2.set_yticks([])
            ax2.set_xlabel('Savings (%)')
            ax2.set_title(f'Efficiency: {label} ({savings_pct:.1f}% savings)', fontweight='bold')
            ax2.text(savings_pct + 2, 0, f'{savings_pct:.1f}%', va='center', fontweight='bold')
            st.pyplot(fig2)
            
        else:
            # Not retrofitted
            retro_df = features_df.copy()
            retro_df['retrofit'] = 1
            raw_retro = model.predict(retro_df)[0]
            retro_pred = scale_prediction(raw_retro)
            
            potential = pred - retro_pred
            potential_pct = (potential / pred) * 100
            potential_conv, _ = convert_energy_unit(potential, unit_option)
            
            m2.metric("💰 Potential Savings", f"{potential_conv:.2f} {unit}", delta=f"{potential_pct:.1f}%")
            m3.metric("🏆 Would Save", f"{potential_pct:.1f}%", delta="If retrofitted")
            
            st.info(f"💡 XGBoost: If retrofitted, save ~{potential_conv:.2f} {unit} ({potential_pct:.1f}%)")
            
            # Simple comparison
            fig_simple, ax_simple = plt.subplots(figsize=(8,5))
            ax_simple.bar(['Current\n(No Retrofit)', 'If Retrofitted'], 
                         [pred, retro_pred], color=['#e74c3c', '#2ecc71'], edgecolor='black')
            for i, v in enumerate([pred, retro_pred]):
                ax_simple.text(i, v + 0.05, f'{v:.2f} kWh', ha='center', fontweight='bold')
            ax_simple.set_ylabel('Energy (kWh)')
            ax_simple.set_title('XGBoost: Potential Retrofit Impact', fontweight='bold')
            ax_simple.grid(axis='y', alpha=0.3)
            st.pyplot(fig_simple)

with col2:
    st.info("""
    **📖 About XGBoost M&V System**
    
    🎯 **Model:** XGBoost Regressor
    
    📊 **Features (15):**
    - Weather (temp, humidity)
    - Time (hour, day, month)
    - Building (area, occupants)
    - Retrofit status
    - Engineered features
    
    🏠 **Malaysia Context:**
    - Scaled for residential homes
    - TNB tariff: ~RM0.52/kWh
    
    ⚡ **vs Random Forest:**
    - XGBoost = sequential learning
    - Usually higher accuracy
    
    💡 **Select 'Yes' to see savings!**
    """)

st.markdown("---")
st.caption("🎓 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
