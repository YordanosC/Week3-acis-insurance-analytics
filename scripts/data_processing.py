import os
import pandas as pd
import json

class DataProcessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def missing_data_summary(self) -> pd.DataFrame:
        """Returns summary of columns with missing data"""
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        missing_percentage = (missing_data / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage (%)': missing_percentage
        })
        return missing_df.sort_values(by='Percentage (%)', ascending=False)
    
    def handle_missing_data(self, missing_type: str, missing_cols: list) -> pd.DataFrame:
        """Handles missing data based on predefined strategies"""
        if missing_type.lower() == 'high':
            self.data = self.data.drop(columns=missing_cols, errors='ignore')
        else:
            for col in missing_cols:
                if col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        self.data[col] = self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown')
                    else:
                        self.data[col] = self.data[col].fillna(self.data[col].median() if not self.data[col].isnull().all() else 0)
        return self.data

if __name__ == "__main__":
    input_file = "Data/interim/MachineLearningRating_v3.txt"
    df = pd.read_csv(input_file, delimiter='|', low_memory=False)
    
    processor = DataProcessing(df)
    missing_summary = processor.missing_data_summary()
    
    with open("Data/processed/missing_data_summary.json", "w") as f:
        json.dump(missing_summary.to_dict(), f)
    
    high_missing = missing_summary[missing_summary['Percentage (%)'] > 50].index.tolist()
    moderate_missing = missing_summary[(missing_summary['Percentage (%)'] <= 50) & (missing_summary['Percentage (%)'] > 20)].index.tolist()
    low_missing = missing_summary[missing_summary['Percentage (%)'] <= 20].index.tolist()
    
    df = processor.handle_missing_data('high', high_missing)
    df = processor.handle_missing_data('moderate', moderate_missing)
    df = processor.handle_missing_data('low', low_missing)
    
    os.makedirs("Data/processed", exist_ok=True)
    df.to_csv("Data/processed/cleaned_data.csv", index=False)