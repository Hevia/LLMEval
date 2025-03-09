import os
import pandas as pd
from SimilarityHelpers import get_levenshtein_distance

def process_csv_files():
    """
    Recursively processes all CSV files in the output/ directory,
    adding a Levenshtein distance column and saving the modified CSV files.
    """
    # Define the root directory
    root_dir = "output/"
    
    # Ensure the root directory exists
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return
    
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Process each CSV file
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                process_single_csv(file_path)
                
def process_single_csv(file_path):
    """
    Processes a single CSV file, adding a Levenshtein distance column and saving it.
    
    Args:
        file_path: Path to the CSV file
    """
    try:
        # Read the CSV file
        print(f"Processing {file_path}...")
        # Use quoting to handle multi-line text fields
        df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
        
        # Check if required columns exist
        if 'Model_Responses' not in df.columns or 'Gold_Labels' not in df.columns:
            print(f"File {file_path} does not have the required columns.")
            return
        
        # Add Levenshtein distance column
        df['levenshtein_distance'] = df.apply(
            lambda row: get_levenshtein_distance(str(row['Model_Responses']), str(row['Gold_Labels'])), 
            axis=1
        )
        
        # Save the modified CSV file, preserving quotes for multi-line text
        df.to_csv(file_path, index=False, quoting=1)
        print(f"Successfully processed {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    process_csv_files()
