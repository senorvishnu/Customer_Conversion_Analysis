import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class DataPreprocessor:
    """
    Data preprocessing class for clickstream customer conversion analysis.
    Handles data loading, cleaning, feature engineering, and preparation for ML models.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_target_variables(self, data):
        """
        Create target variables for classification and regression problems.
        For this project, we'll create synthetic targets based on session behavior.
        """
        # Group by session_id to create session-level features
        session_features = data.groupby('session_id').agg({
            'order': ['count', 'max'],
            'price': ['sum', 'mean', 'max'],
            'page1_main_category': 'nunique',
            'page2_clothing_model': 'nunique',
            'country': 'first',
            'year': 'first',
            'month': 'first',
            'day': 'first'
        }).reset_index()
        
        # Flatten column names
        session_features.columns = ['session_id', 'click_count', 'max_order', 
                                  'total_price', 'avg_price', 'max_price',
                                  'unique_categories', 'unique_products', 
                                  'country', 'year', 'month', 'day']
        
        # Create classification target: 1 if session has high engagement, 0 otherwise
        # High engagement: more than 5 clicks and total price > 50
        session_features['conversion_target'] = (
            (session_features['click_count'] > 5) & 
            (session_features['total_price'] > 50)
        ).astype(int)
        
        # Create regression target: predicted revenue (based on session behavior)
        session_features['revenue_target'] = (
            session_features['total_price'] * 
            (1 + session_features['click_count'] * 0.1) * 
            (1 + session_features['unique_categories'] * 0.05)
        )
        
        return session_features
    
    def engineer_features(self, data):
        """Engineer features from raw clickstream data"""
        # Time-based features
        data['hour'] = np.random.randint(0, 24, len(data))  # Synthetic hour since not in data
        data['day_of_week'] = pd.to_datetime(data[['year', 'month', 'day']]).dt.dayofweek
        
        # Session-level features - simplified approach
        session_stats = data.groupby('session_id').agg({
            'order': 'count',
            'price': ['sum', 'mean', 'std'],
            'page1_main_category': 'nunique',
            'page2_clothing_model': 'nunique',
            'colour': 'nunique',
            'location': 'nunique',
            'model_photography': 'nunique',
            'page': 'nunique',
            'country': 'first',
            'month': 'first',
            'day_of_week': 'first'
        }).reset_index()
        
        # Flatten column names
        session_stats.columns = ['session_id', 'session_length', 'total_spend', 'avg_price', 'price_std',
                               'category_diversity', 'product_diversity', 'color_diversity', 
                               'location_diversity', 'photo_diversity', 'page_diversity', 
                               'country', 'month', 'day_of_week']
        
        # Add derived features with proper handling of division by zero and infinity
        session_stats['avg_price_per_click'] = np.where(
            session_stats['session_length'] > 0,
            session_stats['total_spend'] / session_stats['session_length'],
            0
        )
        session_stats['category_engagement'] = np.where(
            session_stats['session_length'] > 0,
            session_stats['category_diversity'] / session_stats['session_length'],
            0
        )
        session_stats['product_engagement'] = np.where(
            session_stats['session_length'] > 0,
            session_stats['product_diversity'] / session_stats['session_length'],
            0
        )
        
        # Price sensitivity (handle division by zero)
        session_stats['price_sensitivity'] = np.where(
            session_stats['avg_price'] > 0,
            session_stats['price_std'] / session_stats['avg_price'],
            0
        )
        
        # Replace infinity values with finite values
        numerical_cols = ['avg_price_per_click', 'category_engagement', 'product_engagement', 'price_sensitivity']
        for col in numerical_cols:
            if col in session_stats.columns:
                session_stats[col] = session_stats[col].replace([np.inf, -np.inf], 0)
                # Clip extreme values to reasonable range
                session_stats[col] = session_stats[col].clip(-1000, 1000)
        
        # Add min and max price
        price_stats = data.groupby('session_id')['price'].agg(['min', 'max']).reset_index()
        price_stats.columns = ['session_id', 'min_price', 'max_price']
        session_stats = session_stats.merge(price_stats, on='session_id', how='left')
        
        return session_stats
    
    def prepare_features(self, session_data):
        """Prepare features for machine learning models"""
        # Define feature columns
        feature_cols = [
            'session_length', 'total_spend', 'avg_price', 'price_std', 'min_price', 'max_price',
            'category_diversity', 'product_diversity', 'color_diversity', 'location_diversity',
            'photo_diversity', 'page_diversity', 'avg_price_per_click', 'category_engagement',
            'product_engagement', 'price_sensitivity', 'country', 'month', 'day_of_week'
        ]
        
        # Separate numerical and categorical features
        numerical_cols = [col for col in feature_cols if col in session_data.columns and 
                         session_data[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in feature_cols if col in session_data.columns and 
                           session_data[col].dtype == 'object']
        
        self.feature_columns = feature_cols
        self.numerical_columns = numerical_cols
        self.categorical_columns = categorical_cols
        
        # Handle missing values
        for col in numerical_cols:
            if session_data[col].isnull().any():
                session_data[col].fillna(session_data[col].median(), inplace=True)
        
        for col in categorical_cols:
            if session_data[col].isnull().any():
                session_data[col].fillna(session_data[col].mode()[0], inplace=True)
        
        return session_data, numerical_cols, categorical_cols
    
    def create_preprocessing_pipeline(self, numerical_cols, categorical_cols):
        """Create preprocessing pipeline for consistent data transformation"""
        # Numerical features pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def save_preprocessor(self, preprocessor, filepath='models/preprocessor.pkl'):
        """Save the fitted preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Load the fitted preprocessor"""
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            print(f"Preprocessor file not found: {filepath}")
            return None

def main():
    """Main function to demonstrate data preprocessing"""
    preprocessor = DataPreprocessor()
    
    # Load data
    train_data = preprocessor.load_data('train_data.csv')
    test_data = preprocessor.load_data('test_data.csv')
    
    if train_data is not None and test_data is not None:
        # Create session-level features
        train_sessions = preprocessor.engineer_features(train_data)
        test_sessions = preprocessor.engineer_features(test_data)
        
        # Create target variables
        train_targets = preprocessor.create_target_variables(train_data)
        test_targets = preprocessor.create_target_variables(test_data)
        
        # Merge features with targets
        train_final = train_sessions.merge(train_targets[['session_id', 'conversion_target', 'revenue_target']], 
                                         on='session_id', how='left')
        test_final = test_sessions.merge(test_targets[['session_id', 'conversion_target', 'revenue_target']], 
                                       on='session_id', how='left')
        
        # Prepare features
        train_final, num_cols, cat_cols = preprocessor.prepare_features(train_final)
        test_final, _, _ = preprocessor.prepare_features(test_final)
        
        # Save processed data
        train_final.to_csv('processed_train_data.csv', index=False)
        test_final.to_csv('processed_test_data.csv', index=False)
        
        print("Data preprocessing completed successfully!")
        print(f"Train data shape: {train_final.shape}")
        print(f"Test data shape: {test_final.shape}")
        print(f"Numerical features: {len(num_cols)}")
        print(f"Categorical features: {len(cat_cols)}")

if __name__ == "__main__":
    main()
