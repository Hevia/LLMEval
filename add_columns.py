import os
import pandas as pd
from SimilarityHelpers import get_levenshtein_distance

def process_csv_files(force=False):
    """
    Recursively processes all CSV files in the output/ directory,
    adding Levenshtein distance and BERT score columns if they don't already exist.
    
    Args:
        force: If True, will recalculate and overwrite columns even if they already exist
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
                process_single_csv(file_path, force=force)
                
def process_single_csv(file_path, force=False):
    """
    Processes a single CSV file, adding Levenshtein distance and BERT score columns if they don't already exist.
    
    Args:
        file_path: Path to the CSV file
        force: If True, will recalculate and overwrite columns even if they already exist
    """
    try:
        # Read the CSV file with robust error handling
        print(f"Processing {file_path}...")
        try:
            # First try with normal settings but warn for bad lines
            df = pd.read_csv(file_path, quotechar='"', escapechar='\\', on_bad_lines='warn')
        except Exception as e:
            print(f"Error with standard parsing: {str(e)}")
            print("Trying with more permissive CSV parsing...")
            try:
                # Try with more permissive CSV parsing
                df = pd.read_csv(file_path, quotechar='"', escapechar='\\', on_bad_lines='skip')
                print(f"Successfully read file with {len(df)} rows after skipping bad lines")
            except Exception as e2:
                print(f"Error still occurred with permissive parsing: {str(e2)}")
                return
        
        # Check if required columns exist
        if 'Model_Responses' not in df.columns or 'Gold_Labels' not in df.columns:
            print(f"File {file_path} does not have the required columns.")
            return
        
        # Track if we need to save the file
        needs_save = False
        
        # Add Levenshtein distance column if it doesn't exist or if force is True
        if 'levenshtein_distance' not in df.columns or force:
            print(f"Adding levenshtein_distance column to {file_path}")
            df['levenshtein_distance'] = df.apply(
                lambda row: get_levenshtein_distance(str(row['Model_Responses']), str(row['Gold_Labels'])), 
                axis=1
            )
            needs_save = True
        else:
            print(f"levenshtein_distance column already exists in {file_path}")
        
        # Save the modified CSV file only if changes were made
        if needs_save:
            # Save with careful quoting to prevent parsing issues in the future
            df.to_csv(file_path, index=False, quoting=1, escapechar='\\', doublequote=True)
            print(f"Successfully processed and saved {file_path}")
        else:
            print(f"No changes needed for {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    process_csv_files()
    #process_single_csv("./output/XSum/Control/microsoft_Phi-4-mini-instruct_XSum.csv", force=True)
