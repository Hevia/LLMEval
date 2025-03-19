import sys
import os
import argparse
from itertools import product
sys.path.append(os.path.abspath('..'))
from train_classifiers import *
import numpy as np
import pandas as pd

def test_extraction(df: pd.DataFrame) -> None:
    metadata = [
        'Model',
        'Task_Prefix',
        'Dataset_Name',
        'Model_Responses',
        'Gold_Labels'
    ]

    models = [
        "GradientBoosting",
        "LogisticRegression",
        "RandomForest"
    ]
    targets = [
        "prediction"
        # "probability",
        # "status"
    ]

    model_results = ["_".join(p) for p in product(models, targets)]

    # row = df.iloc[0, :]
    problems = 0

    for _, row in df.iterrows():
        for mrx in model_results:
            result_values = set()
            candidates = [c for c in row.keys() if mrx in c]
            # sort candidates by merge recency
            candidates.sort(key=lambda x: (x.count("_"), x), reverse=True)

            for pick in candidates:
                if not np.isnan(row[pick]):
                    result_values.add(row[pick])

            if len(result_values) > 1:
                problems += 1
                print(f"more than one unique value found for {mrx}")
                # print(result_values)
                # print(row)
                # print("{}\t{}\t{}\n".format(pick, row[pick], type(row[pick])))
    print(f"{problems} misalignments")
    return

def extract_classification(df: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        'Model',
        'Task_Prefix',
        'Dataset_Name',
        'Model_Responses',
        'Gold_Labels'
    ]

    models = [
        "GradientBoosting",
        "LogisticRegression",
        "RandomForest"
    ]
    targets = [
        "prediction",
        "probability",
        # "status"
    ]

    model_results = ["_".join(p) for p in product(models, targets)]
            
    # Process and extract results
    results: List[Dict[str, float]] = []

    # Parse the JSON-like structures in Min_K_Responses and rouge_sim_scores
    for _, row in df.iterrows():
        row_results: Dict[str, float] = {}
        # Extract metadata
        for mdx in metadata:
            row_results[mdx] = row[mdx]
        
        for mrx in model_results:
            # get the most recently added column in this type
            # row_results[mrx] = row[mrx]
            candidates = [c for c in row.keys() if mrx in c]
            
            # sort candidates by merge recency
            candidates.sort(key=lambda x: (x.count("_"), x), reverse=True)

            for pick in candidates:
                empty = (row[pick] == None) or (row[pick] == np.nan) or (row[pick] == "") or (row[pick] == "nan")
                if pick in row.keys() and not empty:
                    print(pick.replace(mrx, ""))
                    print(row[pick])
                    print("----")
                    row_results[mrx] = row[pick]
                else:
                    print(f"no value found for {mrx}")
                    print(f"candidates: {candidates}")

            pick = sorted(candidates, reverse=True, key=lambda x: x.count('_')).pop()
            row_results[mrx] = row[pick]

        results.append(row_results)

    results_df = pd.DataFrame(results)
    return results_df

def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numerical metrics from DataFrame.
    Handles the JSON-like structure in Min_K_Responses and rouge_sim_scores.
    Assumes data has been validated.
    
    Args:
        df: DataFrame with validated data
        
    Returns:
        X: Feature matrix
        y: Labels
        metric_names: List of metric names
    """
    # Process and extract metrics
    metrics: List[Dict[str, float]] = []
    metadata = [
        'Model',
        'Task_Prefix',
        'Dataset_Name',
        'Model_Responses',
        'Gold_Labels'
    ]
    # Parse the JSON-like structures in Min_K_Responses and rouge_sim_scores
    for _, row in df.iterrows():
        row_metrics: Dict[str, float] = {}
        
        # Extract metadata
        for mdx in metadata:
            row_metrics[mdx] = row[mdx]

        # Extract cos_sim_scores (should be a valid float at this point)
        if pd.notna(row['cos_sim_scores']):
            row_metrics['cos_sim'] = float(row['cos_sim_scores'])
        else:
            row_metrics['cos_sim'] = 0.0
            
        # Extract levenshtein_distance (should be a valid float at this point)
        if pd.notna(row['levenshtein_distance']):
            row_metrics['levenshtein'] = float(row['levenshtein_distance'])
        else:
            row_metrics['levenshtein'] = 0.0
            
        # Extract Min_K_Responses metrics
        if pd.notna(row['Min_K_Responses']):
            # Convert to string to ensure consistent processing
            min_k_text = str(row['Min_K_Responses'])
            
            # Define regex pattern to extract Min_K values
            # This pattern matches key-value pairs like: 'Min_10.0% Prob': 14.023460388183594
            min_k_pattern = r"'(Min_[\d.]+%\s+Prob)':\s*([\d.]+)"
            
            # Find all matches in the text
            matches = re.findall(min_k_pattern, min_k_text)
            
            # The validation function ensures we have matches here
            for k, v in matches:
                # Clean up the key name
                clean_key = k.replace("'", "").replace("%", "").replace(" ", "_")
                row_metrics[f'min_k_{clean_key}'] = float(v)
        else:
            # Set default values if Min_K_Responses is NA
            row_metrics['min_k_Min_10_Prob'] = 0.0
            row_metrics['min_k_Min_20_Prob'] = 0.0
            row_metrics['min_k_Min_30_Prob'] = 0.0
            row_metrics['min_k_Min_40_Prob'] = 0.0
            row_metrics['min_k_Min_50_Prob'] = 0.0
        
        # Extract rouge_sim_scores metrics
        if pd.notna(row['rouge_sim_scores']):
            # Using regex patterns to extract rouge scores
            patterns = {
                'rouge1': r"rouge1'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)",
                'rouge2': r"rouge2'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)",
                'rougeL': r"rougeL'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)"
            }
            
            rouge_text = str(row['rouge_sim_scores'])
            
            # Extract values using regex for each rouge type
            # The validation function ensures these values are present
            for rouge_type, pattern in patterns.items():
                match = re.search(pattern, rouge_text)
                if match:
                    precision, recall, fmeasure = match.groups()
                    row_metrics[f'{rouge_type}_precision'] = float(precision)
                    row_metrics[f'{rouge_type}_recall'] = float(recall)
                    row_metrics[f'{rouge_type}_fmeasure'] = float(fmeasure)
                else:
                    # This should not happen with validated data, but as a fallback
                    row_metrics[f'{rouge_type}_precision'] = 0.0
                    row_metrics[f'{rouge_type}_recall'] = 0.0
                    row_metrics[f'{rouge_type}_fmeasure'] = 0.0
        else:
            # Set default values if rouge_sim_scores is NA
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                for metric in ['precision', 'recall', 'fmeasure']:
                    row_metrics[f'{rouge_type}_{metric}'] = 0.0
        
        metrics.append(row_metrics)
    
    # Convert to DataFrame
    metric_df: pd.DataFrame = pd.DataFrame(metrics)
    
    # Handle missing values (should be few to none with validation)
    metric_df.fillna(0, inplace=True)
    
    return metric_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_type", type=str)
    args= parser.parse_args()

    # Process CSV files and extract data
    print("Processing CSV files...")
    df: pd.DataFrame = process_csv_files("../output")
    
    # Validate metrics before extraction
    print("\nValidating metrics...")
    validated_df = validate_features(df)
    print(f"Rows after validation: {len(validated_df)} (original: {len(df)})")

    # Fix inconsistent Xsum naming
    validated_df['Dataset_Name'] = validated_df['Dataset_Name'].replace("XSum", "Xsum")

    print(f"\nDataset summary:")
    print(f"Total samples: {validated_df.shape[0]}")

    if args.result_type == "metrics":
     # Extract metrics
        print("\nExtracting metrics...")
        X: pd.DataFrame
        X = extract_metrics(validated_df)

        # Write diff experiment sources to distinct files
        for name, g in X.groupby('Dataset_Name'):
            print(f"Writing validated data for {name}")
            g.to_csv(f"{name}_validated.csv",index=False)
    
    elif args.result_type == "results":
        print("\nExtracting classification results...")
        Z: pd.DataFrame
        # Z = extract_classification(validated_df)
        Z = test_extraction(validated_df)
        # Z.to_csv(f"classification/all_classification.csv", index=False)
        # for name, g in Z.groupby('Dataset_Name'):
        #     print(f"Writing classification results for {name}")
        #     g.to_csv(f"classification/{name}_classification.csv", index=False)
    
    