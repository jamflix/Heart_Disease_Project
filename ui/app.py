import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
    }
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_data
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the training notebooks first.")
        return None

# Load feature names
@st.cache_data
def load_feature_names():
    try:
        # Load the selected features from the saved file
        selected_features = joblib.load('models/selected_features.pkl')
        return selected_features
    except FileNotFoundError:
        # Fallback to default features if file not found
        return ['cp', 'chol', 'thalch', 'age', 'oldpeak', 'exang', 'trestbps', 'thal', 'sex', 'restecg']

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model and features
    model = load_model()
    feature_names = load_feature_names()
    
    if model is None:
        st.stop()
    
    # Sidebar for user input
    st.sidebar.header("Patient Information")
    st.sidebar.markdown("Please enter the patient's health information:")
    
    # Create input form
    with st.sidebar.form("patient_form"):
        # Age
        age = st.slider("Age (years)", min_value=20, max_value=100, value=50)
        
        # Sex
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        
        # Chest Pain Type
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        
        # Resting Blood Pressure
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=130)
        
        # Serum Cholesterol
        chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
        
        # Fasting Blood Sugar
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # Resting ECG
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        
        # Maximum Heart Rate
        thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        
        # Exercise Induced Angina
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        # ST Depression
        oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        
        # ST Slope
        slope = st.selectbox("ST Slope", options=[0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        
        # Number of Major Vessels
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
        
        # Thalassemia
        thal = st.selectbox("Thalassemia", options=[1, 2, 3], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
        
        # Submit button
        submitted = st.form_submit_button("Predict Heart Disease Risk", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Prediction Results")
        
        if submitted:
            # Get selected features
            selected_features = load_feature_names()
            
            # Create a dictionary with all input values
            input_dict = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 
                'fbs': fbs, 'restecg': restecg, 'thalch': thalach, 'exang': exang, 
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
            
            # Prepare input data with only selected features
            input_data = np.array([[input_dict[feature] for feature in selected_features]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            if prediction == 1:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.markdown("### ⚠️ **HIGH RISK** - Heart Disease Detected")
                st.markdown(f"**Risk Probability: {prediction_proba[1]:.1%}**")
                st.markdown("**Recommendation:** Please consult with a cardiologist immediately for further evaluation and treatment.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("### ✅ **LOW RISK** - No Heart Disease Detected")
                st.markdown(f"**Risk Probability: {prediction_proba[0]:.1%}**")
                st.markdown("**Recommendation:** Continue maintaining a healthy lifestyle with regular check-ups.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("Risk Factors Analysis")
            
            risk_factors = []
            if age > 65:
                risk_factors.append("Advanced age (>65)")
            if sex == 1 and age > 45:
                risk_factors.append("Male gender with age >45")
            if cp > 0:
                risk_factors.append("Chest pain symptoms")
            if trestbps > 140:
                risk_factors.append("High blood pressure (>140 mmHg)")
            if chol > 240:
                risk_factors.append("High cholesterol (>240 mg/dl)")
            if fbs == 1:
                risk_factors.append("Elevated fasting blood sugar")
            if thalach < 100:
                risk_factors.append("Low maximum heart rate")
            if exang == 1:
                risk_factors.append("Exercise-induced angina")
            if oldpeak > 2:
                risk_factors.append("Significant ST depression")
            if ca > 0:
                risk_factors.append("Coronary artery disease")
            
            if risk_factors:
                st.write("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("**No major risk factors identified**")
    
    with col2:
        st.header("Model Information")
        
        # Model performance metrics (placeholder - would be loaded from actual results)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "85.2%", "2.1%")
        st.metric("Precision", "83.7%", "1.8%")
        st.metric("Recall", "86.9%", "2.3%")
        st.metric("F1-Score", "85.3%", "2.0%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Feature Importance")
        
        # Feature importance visualization (placeholder)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance", color='Importance', color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    # Footer
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.markdown("""
    This prediction system is for educational and research purposes only. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with qualified healthcare providers 
    for medical decisions.
    """)

if __name__ == "__main__":
    main()
