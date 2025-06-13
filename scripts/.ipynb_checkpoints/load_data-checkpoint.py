import os
import zipfile
import pandas as pd
import io

def load_txt_from_zip(extracted_dir: str, filename: str) -> pd.DataFrame:
    """
    Loads a pipe-separated TXT file from the extracted directory into a pandas DataFrame.
    """
    file_path = os.path.join(extracted_dir, filename)
    df = pd.read_csv(file_path, delimiter='|', low_memory=False)
    return df

def load_data(outer_zip_path: str, filename: str) -> pd.DataFrame:
    try:
        extract_to = "Data/interim/"
        os.makedirs(extract_to, exist_ok=True)
        
        # Extract the zip file directly (no nested handling needed)
        with zipfile.ZipFile(outer_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Load the TXT file from the extracted directory
        df = load_txt_from_zip(extract_to, filename)
        return df
    
    except Exception as e:
        raise RuntimeError(f'Error loading data: {str(e)}')

if __name__ == "__main__":
    zip_path = "Data/raw/MachineLearningRating_v3.zip"
    output_file = "MachineLearningRating_v3.txt"
    df = load_data(zip_path, output_file)
    df.to_csv("Data/interim/MachineLearningRating_v3.txt", sep='|', index=False)