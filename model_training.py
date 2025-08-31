import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Model training class for customer conversion analysis.
    Handles classification, regression, and clustering models.
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.preprocessor = None
        self.feature_columns = None
        
    def load_processed_data(self, train_file='processed_train_data.csv', test_file='processed_test_data.csv'):
        """Load processed data for training"""
        try:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            print(f"Train data loaded: {train_data.shape}")
            print(f"Test data loaded: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None
    
    def prepare_data_for_training(self, data, target_col, feature_cols=None):
        """Prepare data for model training"""
        if feature_cols is None:
            # Exclude target columns and session_id
            exclude_cols = ['session_id', 'conversion_target', 'revenue_target']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinity and extreme values
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            # Replace infinity with large finite values
            X[col] = X[col].replace([np.inf, -np.inf], 1000.0)
            # Clip extreme values to reasonable range
            X[col] = X[col].clip(-1000, 1000)
        
        # Final check for any remaining issues
        if np.isinf(X.values).any():
            print("Warning: Found infinity values in training data. Replacing with finite values...")
            X = X.replace([np.inf, -np.inf], 1000.0)
        
        if (X.values > 1e6).any():
            print("Warning: Found extremely large values in training data. Clipping to reasonable range...")
            X = X.clip(-1000, 1000)
        
        self.feature_columns = feature_cols
        return X, y
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance for classification problems"""
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Original class distribution: {np.bincount(y)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_classification_models(self, X, y, handle_imbalance=True):
        """Train classification models for conversion prediction"""
        print("Training Classification Models...")
        
        if handle_imbalance:
            X, y = self.handle_class_imbalance(X, y, method='smote')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        classification_models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in classification_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
        self.best_models['classification'] = results[best_model_name]['model']
        
        print(f"\nBest classification model: {best_model_name}")
        print(f"Best F1 Score: {results[best_model_name]['metrics']['f1_score']:.4f}")
        
        return results
    
    def train_regression_models(self, X, y):
        """Train regression models for revenue prediction"""
        print("Training Regression Models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        regression_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'svr': SVR()
        }
        
        # Train and evaluate models
        results = {}
        for name, model in regression_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
            
            print(f"{name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
        self.best_models['regression'] = results[best_model_name]['model']
        
        print(f"\nBest regression model: {best_model_name}")
        print(f"Best R² Score: {results[best_model_name]['metrics']['r2']:.4f}")
        
        return results
    
    def train_clustering_models(self, X, n_clusters=3):
        """Train clustering models for customer segmentation"""
        print("Training Clustering Models...")
        
        # Standardize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models
        clustering_models = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in clustering_models.items():
            print(f"Training {name}...")
            
            # Fit model
            clusters = model.fit_predict(X_scaled)
            
            # Calculate metrics (only for KMeans)
            metrics = {}
            if name == 'kmeans':
                metrics['silhouette'] = silhouette_score(X_scaled, clusters)
                metrics['davies_bouldin'] = davies_bouldin_score(X_scaled, clusters)
                metrics['inertia'] = model.inertia_
            
            results[name] = {
                'model': model,
                'clusters': clusters,
                'metrics': metrics,
                'scaler': scaler
            }
            
            if name == 'kmeans':
                print(f"{name} - Silhouette: {metrics['silhouette']:.4f}, Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
            else:
                print(f"{name} - Number of clusters: {len(set(clusters))}")
        
        # Find best model (KMeans is usually preferred for this use case)
        self.best_models['clustering'] = results['kmeans']['model']
        self.best_models['clustering_scaler'] = results['kmeans']['scaler']
        
        print(f"\nBest clustering model: KMeans")
        print(f"Silhouette Score: {results['kmeans']['metrics']['silhouette']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X, y, model_type='classification'):
        """Perform hyperparameter tuning for the best model"""
        print(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == 'classification':
            # Tune Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
            scoring = 'f1_weighted'
            
        elif model_type == 'regression':
            # Tune XGBoost
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=42)
            scoring = 'r2'
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_type, model in self.best_models.items():
            if model_type != 'clustering_scaler':
                filepath = os.path.join(output_dir, f'{model_type}_model.pkl')
                joblib.dump(model, filepath)
                print(f"Saved {model_type} model to {filepath}")
        
        # Save feature columns
        if self.feature_columns:
            feature_file = os.path.join(output_dir, 'feature_columns.pkl')
            joblib.dump(self.feature_columns, feature_file)
            print(f"Saved feature columns to {feature_file}")
    
    def load_models(self, model_dir='models'):
        """Load trained models"""
        model_files = {
            'classification': 'classification_model.pkl',
            'regression': 'regression_model.pkl',
            'clustering': 'clustering_model.pkl'
        }
        
        for model_type, filename in model_files.items():
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                self.best_models[model_type] = joblib.load(filepath)
                print(f"Loaded {model_type} model from {filepath}")
        
        # Load feature columns
        feature_file = os.path.join(model_dir, 'feature_columns.pkl')
        if os.path.exists(feature_file):
            self.feature_columns = joblib.load(feature_file)
            print(f"Loaded feature columns from {feature_file}")

def main():
    """Main function to train all models"""
    trainer = ModelTrainer()
    
    # Load processed data
    train_data, test_data = trainer.load_processed_data()
    
    if train_data is not None and test_data is not None:
        # Train classification models
        X_class, y_class = trainer.prepare_data_for_training(train_data, 'conversion_target')
        classification_results = trainer.train_classification_models(X_class, y_class)
        
        # Train regression models
        X_reg, y_reg = trainer.prepare_data_for_training(train_data, 'revenue_target')
        regression_results = trainer.train_regression_models(X_reg, y_reg)
        
        # Train clustering models
        X_cluster, _ = trainer.prepare_data_for_training(train_data, 'conversion_target')
        clustering_results = trainer.train_clustering_models(X_cluster, n_clusters=4)
        
        # Save models
        trainer.save_models()
        
        print("\nModel training completed successfully!")
        
        # Print summary
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        print("Classification - Best F1 Score:", 
              max(classification_results.values(), key=lambda x: x['metrics']['f1_score'])['metrics']['f1_score'])
        print("Regression - Best R² Score:", 
              max(regression_results.values(), key=lambda x: x['metrics']['r2'])['metrics']['r2'])
        print("Clustering - Silhouette Score:", 
              clustering_results['kmeans']['metrics']['silhouette'])

if __name__ == "__main__":
    main()
