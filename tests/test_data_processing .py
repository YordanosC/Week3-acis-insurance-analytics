# tests/test_data_processing.py
import unittest
import pandas as pd
from scripts.data_processing import DataProcessing  # Import the class from your scripts folder

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """
        This method is run before each test. We set up a sample DataFrame with some missing data.
        """
        # Sample data with some missing values
        self.df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, None, 2, 3],
            'C': [1, 2, 3, 4],
            'D': [None, None, None, None]
        })
        self.data_processing = DataProcessing(self.df)
    
    def test_missing_data_summary(self):
        """
        Test if missing_data_summary returns the correct count and percentage of missing values.
        """
        # Call the missing_data_summary method on the instance
        result = self.data_processing.missing_data_summary()
        
        # Define the expected output
        expected_result = pd.DataFrame({
            'Missing Count': [4, 2, 1],
            'Percentage (%)': [100.0, 50.0, 25.0]
        }, index=['D', 'B', 'A'])

        # Assert that the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result, expected_result)
    
    def test_handle_missing_data(self):
        # Test handling high missing data
        missing_cols_high = ['B', 'D']
        processed_df = self.data_processing.handle_missing_data(missing_type='high', missing_cols=missing_cols_high)
        expected_columns = ['A', 'C']  # Columns B and D should be dropped
        self.assertTrue(all(col in processed_df.columns for col in expected_columns))
        self.assertFalse(any(col in processed_df.columns for col in missing_cols_high))

        # Reset processor for next test
        self.data_processing = DataProcessing(self.df.copy(deep=True))  # Use deep copy

        # Test handling moderate missing data
        missing_cols_moderate = ['A', 'D']
        processed_df = self.data_processing.handle_missing_data(missing_type='moderate', missing_cols=missing_cols_moderate)
        self.assertEqual(processed_df['A'].isnull().sum(), 0)  # No missing values in A
        self.assertEqual(processed_df['D'].isnull().sum(), 0)  # No missing values in D
        # Adjust dtype checks based on actual types in your dataset
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['A'].dtype) or pd.api.types.is_object_dtype(processed_df['A'].dtype))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['D'].dtype) or pd.api.types.is_object_dtype(processed_df['D'].dtype))

        # Reset processor for next test
        self.data_processing = DataProcessing(self.df.copy(deep=True))  # Deep copy again

        # Test default handling (imputation without specifying missing_type)
        missing_cols_default = ['C']
        processed_df = self.data_processing.handle_missing_data(missing_type=None, missing_cols=missing_cols_default)
        self.assertEqual(processed_df['C'].isnull().sum(), 0)  # No missing values in C
        # Adjust dtype check based on expected column type
        self.assertTrue(pd.api.types.is_object_dtype(processed_df['C'].dtype) or pd.api.types.is_numeric_dtype(processed_df['C'].dtype))

if __name__ == '__main__':
    unittest.main()
