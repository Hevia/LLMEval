#!/usr/bin/env python3
# classifier_inference.py - Loads trained classifiers and runs inference on datasets

import os
import glob
import numpy as np
import pandas as pd
import re
import joblib
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.pipeline import Pipeline

# Import necessary functions from train_classifiers.py
from train_classifiers import extract_features, is_string_null_whitespace_or_float


def load_models(model_dir: str = './output/models') -> Dict[str, Pipeline]:
    """
    Load all trained models from the model directory.
    
    Args:
        model_dir: Directory containing the saved models
        
    Returns:
        Dictionary mapping model names to loaded model objects
    """
    models = {}
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # Load all joblib files (models)
    model_files = glob.glob(os.path.join(model_dir, '*.joblib'))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    for model_file in model_files:
        # Extract model name from filename
        model_name = os.path.basename(model_file).replace('.joblib', '')
        
        # Load the model
        print(f"Loading model: {model_name}")
        model = joblib.load(model_file)
        models[model_name] = model
    
    return models


def process_and_predict(models: Dict[str, Pipeline], root_dir: str = './output') -> None:
    """
    Process all CSV files in the given directory structure,
    run inference with each model, and save results back to the CSV files.
    
    Args:
        models: Dictionary of loaded models
        root_dir: Root directory containing the dataset folders
    """
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for dataset in dataset_dirs:
        # Skip the models directory
        if dataset == 'models':
            continue
            
        dataset_path = os.path.join(root_dir, dataset)
        
        # Process Control directory
        control_path = os.path.join(dataset_path, 'Control')
        if os.path.exists(control_path):
            process_directory_files(models, control_path)
        
        # Process Treatment directory
        treatment_path = os.path.join(dataset_path, 'Treatment')
        if os.path.exists(treatment_path):
            process_directory_files(models, treatment_path)


def process_directory_files(models: Dict[str, Pipeline], directory: str) -> None:
    """
    Process all CSV files in a directory, adding model predictions.
    
    Args:
        models: Dictionary of loaded models
        directory: Directory containing CSV files
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Skip if no rows or if Model_Responses column does not exist
        if len(df) == 0 or 'Model_Responses' not in df.columns:
            print(f"  Skipping {csv_file}: No data or missing Model_Responses column")
            continue
        
        # Filter out rows where Model_Responses is null or whitespace
        initial_row_count = len(df)
        df_clean = df[~df['Model_Responses'].apply(is_string_null_whitespace_or_float)]
        filtered_row_count = len(df_clean)
        
        if initial_row_count > filtered_row_count:
            print(f"  Filtered out {initial_row_count - filtered_row_count} rows with empty Model_Responses")
        
        if filtered_row_count == 0:
            print(f"  Skipping {csv_file}: No valid data after filtering")
            continue
        
        # Add a temporary label column (required by extract_features)
        # The actual value doesn't matter as we only need the features
        df_clean['label'] = 0
        
        try:
            # Extract features
            X, _, feature_names = extract_features(df_clean)
            
            # Run inference with each model
            for model_name, model in models.items():
                # Generate predictions
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1
                
                # Create column names
                pred_col = f"{model_name}_prediction"
                prob_col = f"{model_name}_probability"
                
                # Add predictions and probabilities back to the filtered dataframe
                df_clean[pred_col] = predictions
                df_clean[prob_col] = probabilities
                
                # Map predictions to more meaningful labels
                df_clean[f"{model_name}_status"] = df_clean[pred_col].map({
                    0: "Not Contaminated", 
                    1: "Likely Contaminated"
                })

                # Rename the _status column to _label
                df_clean = df_clean.rename(columns={f"{model_name}_status": f"{model_name}_label"})


            
            # Remove temporary label column before merging
            df_clean = df_clean.drop('label', axis=1)
            
            # Merge predictions back into original dataframe
            # Create a temporary index for merging
            df['temp_idx'] = range(len(df))
            df_clean['temp_idx'] = range(len(df_clean))
            
            # Select only the prediction columns for merging
            pred_cols = [col for col in df_clean.columns 
                        if any(model_name in col for model_name in models.keys()) 
                        or col == 'temp_idx']
            
            # Merge predictions back
            df = pd.merge(df, df_clean[pred_cols], on='temp_idx', how='left')
            
            # Remove temporary index
            df = df.drop('temp_idx', axis=1)
            
            # Save updated dataframe back to CSV
            df.to_csv(csv_file, index=False)
            print(f"  Updated {csv_file} with model predictions")
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {str(e)}")
            continue


def main() -> None:
    """Main function to load models and process datasets."""
    try:
        # Load models
        print("Loading models...")
        models = load_models()
        print(f"Loaded {len(models)} models: {', '.join(models.keys())}")
        
        # Process datasets and add predictions
        print("\nProcessing datasets...")
        process_and_predict(models)
        
        print("\nInference complete! All CSV files have been updated with model predictions.")
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())  # Print full traceback for debugging


if __name__ == "__main__":
    main()
