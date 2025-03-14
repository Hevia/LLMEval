import os
import pandas as pd
from SimilarityHelpers import get_levenshtein_distance, get_bert_score_batch

def process_csv_files():
    """
    Recursively processes all CSV files in the output/ directory,
    adding Levenshtein distance and BERT score columns if they don't already exist.
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
    Processes a single CSV file, adding Levenshtein distance and BERT score columns if they don't already exist.
    
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
        
        # Track if we need to save the file
        needs_save = False
        
        # Add Levenshtein distance column if it doesn't exist
        if 'levenshtein_distance' not in df.columns:
            print(f"Adding levenshtein_distance column to {file_path}")
            df['levenshtein_distance'] = df.apply(
                lambda row: get_levenshtein_distance(str(row['Model_Responses']), str(row['Gold_Labels'])), 
                axis=1
            )
            needs_save = True
        else:
            print(f"levenshtein_distance column already exists in {file_path}")
        
        # Add BERT score column if it doesn't exist
        if 'bert_score' not in df.columns:
            print(f"Adding bert_score column to {file_path}")
            
            # Process each row individually since bert-score is computationally intensive
            bert_scores = []
            for i, row in df.iterrows():
                try:
                    gold = str(row['Gold_Labels'])
                    test = str(row['Model_Responses'])
                    
                    # Get BERT score using the function
                    bert_result = get_bert_score_batch([gold], [test])
                    
                    # Extract F1 score from the result
                    if 'f1' in bert_result and len(bert_result['f1']) > 0:
                        bert_scores.append(float(bert_result['f1'][0]))
                    else:
                        print(f"Warning: Could not extract F1 score for row {i}, using 0 as fallback")
                        bert_scores.append(0.0)
                except Exception as e:
                    print(f"Error processing BERT score for row {i}: {str(e)}")
                    bert_scores.append(0.0)
            
            # Add the bert_score column
            df['bert_score'] = bert_scores
            needs_save = True
        else:
            print(f"bert_score column already exists in {file_path}")
        
        # Save the modified CSV file only if changes were made
        if needs_save:
            df.to_csv(file_path, index=False, quoting=1)
            print(f"Successfully processed and saved {file_path}")
        else:
            print(f"No changes needed for {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    process_csv_files()
