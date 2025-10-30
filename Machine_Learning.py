import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class BusCaptainRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.best_model = None
        
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        print("Preparing features for machine learning...")
        
        # Create a copy to avoid modifying original data
        df_ml = df.copy()
        
        # Basic features from your existing analysis
        features = ['total_alarms']
        
        # Add average speed if available
        if 'avg_speed' in df_ml.columns:
            features.append('avg_speed')
            # Fill missing speeds with median
            df_ml['avg_speed'] = df_ml['avg_speed'].fillna(df_ml['avg_speed'].median())
        
        # Extract features from routes information
        if 'routes' in df_ml.columns:
            # Number of different routes per driver
            df_ml['num_routes'] = df_ml['routes'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
            features.append('num_routes')
        
        # Prepare feature matrix and target
        X = df_ml[features].copy()
        y = df_ml['has_accident']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=features, index=df_ml.index)
        
        print(f"Features used: {features}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, features
    
    def train_models(self, df):
        """Train multiple models and select the best one"""
        X, y, features = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models to try
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, class_weight='balanced')
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Use AUC-ROC as evaluation metric
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            if name == 'random_forest':
                self.feature_importance[name] = dict(zip(features, model.feature_importances_))
            
            self.models[name] = {
                'model': model,
                'auc_score': auc_score
            }
            
            print(f"{name.replace('_', ' ').title()}: AUC = {auc_score:.4f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
        
        self.best_model = best_model_name
        print(f"\nBest model: {best_model_name} with AUC = {best_score:.4f}")
        
        return X_test, y_test, best_model_name
    
    def predict_risk(self, df, model_name=None):
        """Predict risk scores for all drivers"""
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        X, _, _ = self.prepare_features(df)
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Scale features if using logistic regression
        if model_name == 'logistic_regression':
            X_processed = self.scalers['standard'].transform(X)
        else:
            X_processed = X
        
        # Get probability predictions
        risk_scores = model.predict_proba(X_processed)[:, 1]
        
        # Create results dataframe
        results = df[['driver', 'total_alarms', 'has_accident']].copy()
        results['risk_score'] = risk_scores
        
        # Classify drivers into risk categories
        results['risk_category'] = pd.cut(
            results['risk_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        return results
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest model"""
        if 'random_forest' not in self.feature_importance:
            print("Random Forest feature importance not available")
            return
        
        importance_dict = self.feature_importance['random_forest']
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances, y=features, palette='viridis', ax=ax)
        ax.set_title('Feature Importance - Random Forest')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        return fig
    
    def plot_risk_distribution(self, risk_results):
        """Plot distribution of risk scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk score distribution
        sns.histplot(risk_results['risk_score'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Distribution of Risk Scores')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Number of Drivers')
        
        # Risk category distribution
        risk_counts = risk_results['risk_category'].value_counts()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette='RdYlGn_r', ax=ax2)
        ax2.set_title('Drivers by Risk Category')
        ax2.set_xlabel('Risk Category')
        ax2.set_ylabel('Number of Drivers')
        
        plt.tight_layout()
        return fig
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """Comprehensive model evaluation"""
        if model_name is None:
            model_name = self.best_model
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        if model_name == 'logistic_regression':
            X_test_processed = self.scalers['standard'].transform(X_test)
        else:
            X_test_processed = X_test
        
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Print classification report
        print("\n" + "="*50)
        print(f"EVALUATION RESULTS - {model_name.upper()}")
        print("="*50)
        print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        ax2.plot(recall, precision, marker='.')
        ax2.set_title('Precision-Recall Curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filename='bus_risk_predictor.joblib'):
        """Save the trained model"""
        if self.best_model:
            model_data = {
                'best_model': self.models[self.best_model]['model'],
                'best_model_name': self.best_model,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_data, filename)
            print(f"Model saved as {filename}")
    
    def load_model(self, filename='bus_risk_predictor.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.models = {model_data['best_model_name']: {'model': model_data['best_model']}}
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.best_model = model_data['best_model_name']
        print(f"Model loaded from {filename}")

def run_machine_learning_analysis():
    """Main function to run ML analysis"""
    print("Starting Machine Learning Analysis for At-Risk Bus Captains")
    print("="*60)
    
    # Load the combined data
    try:
        df = pd.read_parquet("Driver_Summary_Combined.parquet")
        print(f"Loaded data with {len(df)} drivers")
    except FileNotFoundError:
        print("Error: Driver_Summary_Combined.parquet not found. Run Analyze.py first.")
        return None
    
    # Initialize and train the predictor
    predictor = BusCaptainRiskPredictor()
    
    # Train models
    X_test, y_test, best_model = predictor.train_models(df)
    
    # Get risk predictions
    risk_results = predictor.predict_risk(df)
    
    # Display top at-risk drivers
    top_risky = risk_results.sort_values('risk_score', ascending=False).head(10)
    print("\n" + "="*50)
    print("TOP 10 AT-RISK BUS CAPTAINS")
    print("="*50)
    print(top_risky[['driver', 'total_alarms', 'has_accident', 'risk_score', 'risk_category']].to_string(index=False))
    
    # Save results
    risk_results.to_parquet('driver_risk_assessment.parquet', index=False)
    print(f"\nRisk assessment saved to 'driver_risk_assessment.parquet'")
    
    # Save model
    predictor.save_model()
    
    return predictor, risk_results, X_test, y_test

if __name__ == "__main__":
    run_machine_learning_analysis()
