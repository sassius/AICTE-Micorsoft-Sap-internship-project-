# backend/model_training.py
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    @staticmethod
    def train_models(X_train, X_test, y_train, y_test, disease_name):
        """
        Train multiple machine learning models for medical diagnosis
        
        Parameters:
        -----------
        X_train : array-like
            Training feature data
        X_test : array-like
            Testing feature data
        y_train : array-like
            Training target data
        y_test : array-like
            Testing target data
        disease_name : str
            Name of the disease for model saving
        
        Returns:
        --------
        dict: Trained models with their performance metrics
        """
        # Initialize models
        models = {
            'SVM': SVC(probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        
        # Train and evaluate models
        results = {}
        best_model = None
        best_accuracy = 0
        
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report
            }
            
            # Track the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
            
            # Save the model
            joblib.dump(model, f'models/{disease_name.lower().replace(" ", "_")}_{name.lower().replace(" ", "_")}_model.pkl')
        
        return {
            'results': results,
            'best_model': best_model,
            'best_accuracy': best_accuracy
        }