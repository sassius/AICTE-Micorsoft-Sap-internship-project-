# backend/prediction_utils.py
import joblib
import numpy as np

class PredictionUtils:
    @staticmethod
    def load_model(model_path):
        """
        Load a pre-trained machine learning model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        
        Returns:
        --------
        Loaded machine learning model
        """
        return joblib.load(model_path)
    
    @staticmethod
    def predict_disease(model, scaler, input_features):
        """
        Make disease prediction using a trained model
        
        Parameters:
        -----------
        model : sklearn model
            Trained machine learning model
        scaler : sklearn scaler
            Feature scaler used during training
        input_features : list
            Input features for prediction
        
        Returns:
        --------
        tuple: (prediction, prediction_probability)
        """
        # Prepare input for prediction
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale the input
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        
        return prediction[0], prediction_proba[0][1]