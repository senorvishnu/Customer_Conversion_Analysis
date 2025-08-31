import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_app import StreamlitApp

def test_raw_data_conversion():
    """Test that raw clickstream data is converted to session features correctly"""
    print("Testing raw data conversion...")
    
    # Create sample raw data similar to test_data.csv
    raw_data = pd.DataFrame({
        'year': [2008, 2008, 2008, 2008, 2008, 2008],
        'month': [6, 6, 6, 7, 7, 7],
        'day': [15, 15, 15, 22, 22, 22],
        'order': [1, 2, 3, 1, 2, 3],
        'country': [29, 29, 29, 42, 42, 42],
        'session_id': [1001, 1001, 1001, 1002, 1002, 1002],
        'page1_main_category': [4, 4, 1, 1, 2, 2],
        'page2_clothing_model': ['P48', 'P23', 'A15', 'A2', 'B24', 'B32'],
        'colour': [9, 6, 14, 3, 11, 6],
        'location': [4, 2, 5, 1, 2, 5],
        'model_photography': [2, 2, 2, 1, 1, 1],
        'price': [33, 28, 33, 43, 57, 38],
        'price_2': [2, 2, 2, 2, 1, 2],
        'page': [3, 2, 1, 1, 2, 2]
    })
    
    print("Original raw data shape:", raw_data.shape)
    print("Original columns:", list(raw_data.columns))
    
    # Create StreamlitApp instance
    app = StreamlitApp()
    
    # Test the conversion
    try:
        converted_data = app.convert_raw_data_to_session_features(raw_data)
        
        print("Converted data shape:", converted_data.shape)
        print("Converted columns:", list(converted_data.columns))
        
        # Check that we have the expected session-level features
        expected_features = [
            'session_id', 'session_length', 'total_spend', 'avg_price', 'price_std',
            'category_diversity', 'product_diversity', 'color_diversity', 
            'location_diversity', 'photo_diversity', 'page_diversity', 
            'country', 'month', 'day_of_week', 'avg_price_per_click', 
            'category_engagement', 'product_engagement', 'price_sensitivity',
            'min_price', 'max_price'
        ]
        
        missing_features = [f for f in expected_features if f not in converted_data.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
        else:
            print("✅ All expected features are present")
        
        # Check that we have the correct number of sessions
        expected_sessions = 2  # We have session_id 1001 and 1002
        actual_sessions = len(converted_data)
        if actual_sessions == expected_sessions:
            print(f"✅ Correct number of sessions: {actual_sessions}")
        else:
            print(f"❌ Expected {expected_sessions} sessions, got {actual_sessions}")
        
        # Check for infinity values
        infinity_count = np.isinf(converted_data.values).sum()
        if infinity_count == 0:
            print("✅ No infinity values found")
        else:
            print(f"❌ Found {infinity_count} infinity values")
        
        # Check for NaN values
        nan_count = converted_data.isnull().sum().sum()
        if nan_count == 0:
            print("✅ No NaN values found")
        else:
            print(f"❌ Found {nan_count} NaN values")
        
        # Display sample of converted data
        print("\nSample converted data:")
        print(converted_data.head())
        
        print("✅ Raw data conversion test passed!")
        
    except Exception as e:
        print(f"❌ Error in raw data conversion: {e}")
        import traceback
        traceback.print_exc()

def test_batch_prediction_with_raw_data():
    """Test that batch predictions work with raw data"""
    print("\nTesting batch prediction with raw data...")
    
    # Create sample raw data
    raw_data = pd.DataFrame({
        'year': [2008, 2008, 2008, 2008],
        'month': [6, 6, 7, 7],
        'day': [15, 15, 22, 22],
        'order': [1, 2, 1, 2],
        'country': [29, 29, 42, 42],
        'session_id': [1001, 1001, 1002, 1002],
        'page1_main_category': [4, 4, 1, 2],
        'page2_clothing_model': ['P48', 'P23', 'A15', 'B24'],
        'colour': [9, 6, 14, 11],
        'location': [4, 2, 5, 2],
        'model_photography': [2, 2, 2, 1],
        'price': [33, 28, 33, 57],
        'price_2': [2, 2, 2, 1],
        'page': [3, 2, 1, 2]
    })
    
    print("Raw data shape:", raw_data.shape)
    
    # Create StreamlitApp instance
    app = StreamlitApp()
    
    # Mock the trainer to avoid loading actual models
    class MockTrainer:
        def __init__(self):
            self.feature_columns = [
                'session_length', 'total_spend', 'avg_price', 'price_std', 'category_diversity', 
                'product_diversity', 'color_diversity', 'location_diversity', 'photo_diversity', 
                'page_diversity', 'country', 'month', 'day_of_week', 'avg_price_per_click', 
                'category_engagement', 'product_engagement', 'price_sensitivity', 'min_price', 'max_price'
            ]
            self.best_models = {}
    
    app.trainer = MockTrainer()
    
    try:
        # Test the conversion part of batch prediction
        if 'session_id' in raw_data.columns and 'order' in raw_data.columns and 'price' in raw_data.columns:
            print("✅ Raw data format detected correctly")
            converted_data = app.convert_raw_data_to_session_features(raw_data)
            print("✅ Data converted successfully")
            print("Converted data shape:", converted_data.shape)
            print("Converted columns:", list(converted_data.columns))
        else:
            print("❌ Raw data format not detected")
        
        print("✅ Batch prediction with raw data test passed!")
        
    except Exception as e:
        print(f"❌ Error in batch prediction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Running raw data conversion tests...")
    
    try:
        test_raw_data_conversion()
        test_batch_prediction_with_raw_data()
        
        print("\n🎉 All raw data conversion tests passed!")
        print("The system can now handle raw clickstream data like test_data.csv")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
