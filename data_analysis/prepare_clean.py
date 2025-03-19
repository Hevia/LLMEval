import ast
import pandas as pd
import re
from ast import literal_eval # this will evaluate a string as literal Python objs
import argparse
from glob import glob
import os

def write_to_dir(f, source, group, df, result_type):
    result_filename = f'{result_type}_{(f.split("/")[-1])}'
    result_dir = f'{result_type}/{source}/{group}'
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, result_filename)
    df.to_csv(file_path)
    return

def drop_blank_responses(df):
    """Log number of blank model responses before dropping those rows from the final data."""
    blank = int(df.Model_Responses.isna().sum())
    print("Dropping {} blank outputs...".format(blank))
    return df.dropna(subset=["Model_Responses"])

def parse_rouge_scores(rouge_str):
    """Convert a stringified ROUGE score dictionary into a proper dictionary."""
    try:
        # Extract all 'Score(...)' parts using regex
        scores = re.findall(r"Score\(precision=([\d\.]+), recall=([\d\.]+), fmeasure=([\d\.]+)\)", rouge_str)
       
        if not scores:  # If no matches, return None
            return None

        # Extract values for each ROUGE type
        rouge_keys = ['rouge1', 'rouge2', 'rougeL']
        rouge_dict = {rouge_keys[i]: {
            "precision": float(scores[i][0]),
            "recall": float(scores[i][1]),
            "f1": float(scores[i][2])
        } for i in range(len(scores))}

        return rouge_dict
    except Exception as e:
        print(f"Error parsing: {rouge_str} -> {e}")  
        return None  # Handle errors gracefully
    
def safe_literal_eval(val):
    """Safely converts a string to a dictionary while handling NaN values."""
    if isinstance(val, str):  # Ensure it's a string
        try:
            # Replace 'nan' (string) with 'None' so that it can be parsed correctly
            cleaned_val = val.replace("nan", "None")
            return ast.literal_eval(cleaned_val)  # Convert to dictionary
        except (SyntaxError, ValueError):
            print(f"Skipping invalid entry: {val}")  # Debugging info
            return None  # Return None if conversion fails
    return val  # Return as-is if not a string

def process(wildcard="./validated/*prediction.csv"):
    files = glob(wildcard)

    for f in files:
        print(f"Parsing {f}...")
        df = pd.read_csv(f, dtype=object)

        df["Min_K_Responses"] = df["Min_K_Responses"].apply(safe_literal_eval)

        # Verify the conversion
        assert isinstance(df["Min_K_Responses"][0], dict)
                                        
        df['rouge_sim_scores'] = df['rouge_sim_scores'].apply(parse_rouge_scores)
        df['cos_sim_scores'] = df['cos_sim_scores'].apply(float)
        df['levenshtein_distance'] = df['levenshtein_distance'].apply(float)


        # Apply parsing to each dictionary entry
        df = df.join(df['rouge_sim_scores'].apply(lambda x: pd.Series({
            'rouge1_precision': x['rouge1']['precision'],
            'rouge1_recall': x['rouge1']['recall'],
            'rouge1_f1': x['rouge1']['f1'],
            'rouge2_precision': x['rouge2']['precision'],
            'rouge2_recall': x['rouge2']['recall'],
            'rouge2_f1': x['rouge2']['f1'],
            'rougeL_precision': x['rougeL']['precision'],
            'rougeL_recall': x['rougeL']['recall'],
            'rougeL_f1': x['rougeL']['f1']
        })))


        df = df.join(df['Min_K_Responses'].apply(lambda x: pd.Series({
            'Min_10.0%_Prob': x['Min_10.0% Prob'],
            'Min_20.0%_Prob': x['Min_20.0% Prob'],
            'Min_30.0%_Prob': x['Min_30.0% Prob'],
            'Min_40.0%_Prob': x['Min_40.0% Prob'],
            'Min_50.0%_Prob': x['Min_50.0% Prob']
        })))

        # Drop old score columns
        df = df.drop(columns=['rouge_sim_scores', 'Min_K_Responses'])

        # Drop blank outputs
        df = drop_blank_responses(df)

        dataset = f.split("/")[-1].split(".")[0].split("_")[0]
        df.to_csv(f"./validated/{dataset}_scores_predictions.csv")

def run(source, group):
    input_files = glob(f"../output/{source}/{group}/*.csv")
    # input_files = glob(f"./validated/*prediction.csv")
    
    for f in input_files:
        print(f"Parsing {f}...")
        df = pd.read_csv(f, dtype=object)

        df["Min_K_Responses"] = df["Min_K_Responses"].apply(safe_literal_eval)

        # Verify the conversion
        assert isinstance(df["Min_K_Responses"][0], dict)
                                        
        df['rouge_sim_scores'] = df['rouge_sim_scores'].apply(parse_rouge_scores)
        df['cos_sim_scores'] = df['cos_sim_scores'].apply(float)
        df['levenshtein_distance'] = df['levenshtein_distance'].apply(float)


        # Apply parsing to each dictionary entry
        df = df.join(df['rouge_sim_scores'].apply(lambda x: pd.Series({
            'rouge1_precision': x['rouge1']['precision'],
            'rouge1_recall': x['rouge1']['recall'],
            'rouge1_f1': x['rouge1']['f1'],
            'rouge2_precision': x['rouge2']['precision'],
            'rouge2_recall': x['rouge2']['recall'],
            'rouge2_f1': x['rouge2']['f1'],
            'rougeL_precision': x['rougeL']['precision'],
            'rougeL_recall': x['rougeL']['recall'],
            'rougeL_f1': x['rougeL']['f1']
        })))


        df = df.join(df['Min_K_Responses'].apply(lambda x: pd.Series({
            'Min_10.0%_Prob': x['Min_10.0% Prob'],
            'Min_20.0%_Prob': x['Min_20.0% Prob'],
            'Min_30.0%_Prob': x['Min_30.0% Prob'],
            'Min_40.0%_Prob': x['Min_40.0% Prob'],
            'Min_50.0%_Prob': x['Min_50.0% Prob']
        })))

        # Drop old score columns
        df = df.drop(columns=['rouge_sim_scores', 'Min_K_Responses'])

        # Drop blank outputs
        df = drop_blank_responses(df)

        # Classifier strings and column lists
        classifiers = ["GradientBoosting", "LogisticRegression", "RandomForest"]
        classifier_cols = [col for col in df.columns if any([cx in col for cx in classifiers])]
        core = ['Model', 'Task_Prefix', 'Dataset_Name', 'Model_Responses','Gold_Labels']

        # Split scores from classifier results
        df_scores = df.drop(columns=classifier_cols)
        df_classifiers = df[core + classifier_cols]

        # Write to output
        write_to_dir(f, source, group, df_scores, "scores")
        write_to_dir(f, source, group, df_classifiers, "classifiers")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_data", type=str)
    parser.add_argument("model_group", type=str)
    
    args= parser.parse_args()

    run(args.source_data, args.model_group)
        


