#!/usr/bin/env python3
"""
Complete pipeline runner for Customer Conversion Analysis project.
This script processes the data, trains models, and provides a summary of results.
"""

import os
import sys
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_name):
    """Print a step indicator"""
    print(f"\n🔧 {step_name}...")

def print_success(message):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message"""
    print(f"❌ {message}")

def main():
    """Main pipeline runner"""
    start_time = time.time()
    
    print_header("Customer Conversion Analysis Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Preprocessing
        print_step("Data Preprocessing")
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Load data
        train_data = preprocessor.load_data('train_data.csv')
        test_data = preprocessor.load_data('test_data.csv')
        
        if train_data is None or test_data is None:
            print_error("Failed to load data files")
            return
        
        print_success(f"Loaded train data: {train_data.shape}")
        print_success(f"Loaded test data: {test_data.shape}")
        
        # Create session-level features
        print_step("Feature Engineering")
        train_sessions = preprocessor.engineer_features(train_data)
        test_sessions = preprocessor.engineer_features(test_data)
        
        print_success("Feature engineering completed")
        
        # Create target variables
        print_step("Target Variable Creation")
        train_targets = preprocessor.create_target_variables(train_data)
        test_targets = preprocessor.create_target_variables(test_data)
        
        print_success("Target variables created")
        
        # Merge features with targets
        train_final = train_sessions.merge(
            train_targets[['session_id', 'conversion_target', 'revenue_target']], 
            on='session_id', how='left'
        )
        test_final = test_sessions.merge(
            test_targets[['session_id', 'conversion_target', 'revenue_target']], 
            on='session_id', how='left'
        )
        
        # Prepare features
        train_final, num_cols, cat_cols = preprocessor.prepare_features(train_final)
        test_final, _, _ = preprocessor.prepare_features(test_final)
        
        # Save processed data
        train_final.to_csv('processed_train_data.csv', index=False)
        test_final.to_csv('processed_test_data.csv', index=False)
        
        print_success(f"Processed train data saved: {train_final.shape}")
        print_success(f"Processed test data saved: {test_final.shape}")
        print_success(f"Numerical features: {len(num_cols)}")
        print_success(f"Categorical features: {len(cat_cols)}")
        
        # Step 2: Model Training
        print_step("Model Training")
        from model_training import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Train classification models
        print_step("Training Classification Models")
        X_class, y_class = trainer.prepare_data_for_training(train_final, 'conversion_target')
        classification_results = trainer.train_classification_models(X_class, y_class)
        
        # Find best classification model
        best_class_model = max(classification_results.keys(), 
                              key=lambda x: classification_results[x]['metrics']['f1_score'])
        best_class_f1 = classification_results[best_class_model]['metrics']['f1_score']
        
        print_success(f"Best classification model: {best_class_model} (F1: {best_class_f1:.4f})")
        
        # Train regression models
        print_step("Training Regression Models")
        X_reg, y_reg = trainer.prepare_data_for_training(train_final, 'revenue_target')
        regression_results = trainer.train_regression_models(X_reg, y_reg)
        
        # Find best regression model
        best_reg_model = max(regression_results.keys(), 
                            key=lambda x: regression_results[x]['metrics']['r2'])
        best_reg_r2 = regression_results[best_reg_model]['metrics']['r2']
        
        print_success(f"Best regression model: {best_reg_model} (R²: {best_reg_r2:.4f})")
        
        # Train clustering models
        print_step("Training Clustering Models")
        X_cluster, _ = trainer.prepare_data_for_training(train_final, 'conversion_target')
        clustering_results = trainer.train_clustering_models(X_cluster, n_clusters=4)
        
        clustering_silhouette = clustering_results['kmeans']['metrics']['silhouette']
        print_success(f"Best clustering model: K-Means (Silhouette: {clustering_silhouette:.4f})")
        
        # Save models
        print_step("Saving Models")
        trainer.save_models()
        print_success("All models saved successfully")
        
        # Step 3: Summary
        print_header("Pipeline Summary")
        
        # Data summary
        print("📊 Data Summary:")
        print(f"   - Original train data: {train_data.shape}")
        print(f"   - Original test data: {test_data.shape}")
        print(f"   - Processed train data: {train_final.shape}")
        print(f"   - Processed test data: {test_final.shape}")
        print(f"   - Features engineered: {len(num_cols) + len(cat_cols)}")
        
        # Model performance summary
        print("\n🤖 Model Performance:")
        print(f"   - Classification: {best_class_model} (F1: {best_class_f1:.4f})")
        print(f"   - Regression: {best_reg_model} (R²: {best_reg_r2:.4f})")
        print(f"   - Clustering: K-Means (Silhouette: {clustering_silhouette:.4f})")
        
        # Conversion statistics
        conversion_rate = (train_final['conversion_target'].sum() / len(train_final)) * 100
        avg_revenue = train_final['revenue_target'].mean()
        
        print("\n📈 Business Metrics:")
        print(f"   - Conversion rate: {conversion_rate:.2f}%")
        print(f"   - Average revenue per session: ${avg_revenue:.2f}")
        print(f"   - Total sessions: {len(train_final):,}")
        
        # Timing
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ Pipeline completed in {duration:.2f} seconds")
        print_success("Pipeline completed successfully!")
        
        # Next steps
        print_header("Next Steps")
        print("🚀 To run the Streamlit application:")
        print("   streamlit run streamlit_app.py")
        print("\n📁 Files created:")
        print("   - processed_train_data.csv")
        print("   - processed_test_data.csv")
        print("   - models/classification_model.pkl")
        print("   - models/regression_model.pkl")
        print("   - models/clustering_model.pkl")
        print("   - models/feature_columns.pkl")
        
    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
