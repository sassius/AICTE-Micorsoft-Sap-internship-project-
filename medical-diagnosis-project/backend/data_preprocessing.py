# backend/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    @staticmethod
    def preprocess_data(data_path, target_column):
        """
        Preprocess medical dataset for machine learning models
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file
        target_column : str
            Name of the target column for prediction
        
        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Load the data
        df = pd.read_csv(data_path)
        
        # Handle missing values
        df.fillna(df.median(), inplace=True)
        
        # Encode categorical variables
        for column in df.select_dtypes(include=['object']).columns:
            if column != target_column:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler