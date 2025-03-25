# frontend/app.py
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_preprocessing import DataPreprocessor
from backend.model_training import ModelTrainer
from backend.prediction_utils import PredictionUtils

class MedicalDiagnosisApp:
    def __init__(self):
        """
        Initialize the Medical Diagnosis Application
        """
        # Set page configuration
        st.set_page_config(
            page_title="MediPredict AI", 
            page_icon="🩺", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for styling
        self.add_custom_css()
        
        # Initialize session state for models and scalers
        if 'models' not in st.session_state:
            st.session_state.models = {}
        if 'scalers' not in st.session_state:
            st.session_state.scalers = {}

    def add_custom_css(self):
        """
        Add custom CSS for enhanced styling
        """
        st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    def train_models(self):
        """
        Train models for different diseases
        """
        st.sidebar.header("🧠 Model Training")
        
        # List of diseases and their data paths
        diseases = {
            'Heart Disease': 'data/heart_disease_data.csv',
            'Diabetes': 'data/diabetes_data.csv',
            # Add more diseases as needed
        }
        
        # Training progress
        progress_bar = st.sidebar.progress(0)
        
        for i, (disease, data_path) in enumerate(diseases.items()):
            st.sidebar.write(f"Training {disease} Model...")
            
            try:
                # Preprocess data
                X_train, X_test, y_train, y_test, scaler = DataPreprocessor.preprocess_data(
                    data_path, 'target'
                )
                
                # Train models
                training_results = ModelTrainer.train_models(
                    X_train, X_test, y_train, y_test, disease
                )
                
                # Store in session state
                st.session_state.models[disease] = training_results['best_model']
                st.session_state.scalers[disease] = scaler
                
                # Update progress
                progress_bar.progress((i + 1) / len(diseases))
                
                st.sidebar.success(f"{disease} Model Trained (Accuracy: {training_results['best_accuracy']:.2%})")
            
            except Exception as e:
                st.sidebar.error(f"Error training {disease} model: {str(e)}")
        
        st.sidebar.success("🎉 All Models Trained Successfully!")

    def disease_input_section(self, selected_disease):
        """
        Collect input features for disease prediction
        """
        st.header(f"🔍 {selected_disease} Diagnosis")
        
        # Input collection based on disease
        if selected_disease == "Heart Disease":
            return self.heart_disease_inputs()
        elif selected_disease == "Diabetes":
            return self.diabetes_inputs()
        # Add more disease-specific input methods
        
    def heart_disease_inputs(self):
        """
        Collect heart disease specific inputs
        """
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, value=40)
            sex = st.selectbox('Sex', ['Male', 'Female'])
            chest_pain = st.selectbox('Chest Pain Type', [
                'Typical Angina', 'Atypical Angina', 
                'Non-Anginal Pain', 'Asymptomatic'
            ])
        
        with col2:
            blood_sugar = st.number_input('Fasting Blood Sugar', min_value=0.0, value=100.0)
            rest_bp = st.number_input('Resting Blood Pressure', min_value=0, value=120)
            cholesterol = st.number_input('Cholesterol', min_value=0, value=200)
        
        # Convert categorical inputs
        sex = 1 if sex == 'Male' else 0
        chest_pain_map = {
            'Typical Angina': 0, 
            'Atypical Angina': 1, 
            'Non-Anginal Pain': 2, 
            'Asymptomatic': 3
        }
        chest_pain = chest_pain_map[chest_pain]
        
        return [age, sex, chest_pain, rest_bp, cholesterol, blood_sugar]

    def diabetes_inputs(self):
        """
        Collect diabetes specific inputs
        """
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, value=0)
            glucose = st.number_input('Glucose Level', min_value=0, value=100)
            blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
        
        with col2:
            skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
            insulin = st.number_input('Insulin Level', min_value=0, value=0)
            bmi = st.number_input('BMI', min_value=0.0, value=25.0)
        
        return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi]

    def prediction_section(self, selected_disease, input_features):
        """
        Perform disease prediction
        """
        if st.button("🩺 Get Diagnosis Prediction"):
            # Check if model is trained
            if selected_disease not in st.session_state.models:
                st.warning("Please train the model first!")
                return
            
            # Get model and scaler from session state
            model = st.session_state.models[selected_disease]
            scaler = st.session_state.scalers[selected_disease]
            
            # Make prediction
            prediction, probability = PredictionUtils.predict_disease(
                model, scaler, input_features
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Diagnosis Result", 
                    "High Risk" if prediction == 1 else "Low Risk"
                )
            
            with col2:
                st.metric(
                    "Risk Probability", 
                    f"{probability*100:.2f}%"
                )
            
            # Detailed interpretation
            st.markdown("""
            ### 📋 Detailed Interpretation
            - **Risk Assessment**: Probabilistic evaluation of disease risk
            - **Recommendation**: Consult healthcare professional for comprehensive diagnosis
            - **Disclaimer**: This is a screening tool, not a definitive medical diagnosis
            """)

    def run(self):
        """
        Main application runner
        """
        # Title and introduction
        st.title("🩺 MediPredict AI: Smart Medical Diagnosis System")
        
        # Sidebar controls
        st.sidebar.header("🧠 Disease Prediction")
        diseases = ['Heart Disease', 'Diabetes']
        selected_disease = st.sidebar.selectbox(
            "Select Disease for Diagnosis", 
            diseases
        )
        
        # Model training button
        if st.sidebar.button("🚀 Train Models"):
            self.train_models()
        
        # Disease input section
        input_features = self.disease_input_section(selected_disease)
        
        # Prediction section
        self.prediction_section(selected_disease, input_features)

def main():
    app = MedicalDiagnosisApp()
    app.run()

if __name__ == '__main__':
    main()