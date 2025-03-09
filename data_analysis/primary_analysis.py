import ast
import pandas as pd
import re
<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval # this will evaluate a string as literal Python objs
import argparse
=======
from ast import literal_eval # this will evaluate a string as literal Python objs
import argparse
from glob import glob
import os
>>>>>>> origin/jessica-dev

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("file_dir", type=str)
    
    args= parser.parse_args()

    input_dir = f'/Users/chenxinliu/LLMEval/output/{args.file_dir}'
    output_dir = f'data_{args.file_dir}'
    df = pd.read_csv(input_dir, dtype=object)


    df["Min_K_Responses"] = df["Min_K_Responses"].apply(safe_literal_eval)

    # Verify the conversion
    print(type(df["Min_K_Responses"][0]))  # Should print <class 'dict'>
                                    
    df['rouge_sim_scores'] = df['rouge_sim_scores'].apply(parse_rouge_scores)
    df['cos_sim_scores'] = df['cos_sim_scores'].apply(float)


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


    df = df.drop(columns=['rouge_sim_scores', 'Min_K_Responses'])

    df.to_csv(output_dir)
=======
    parser.add_argument("source_data", type=str)
    parser.add_argument("model_group", type=str)
    
    args= parser.parse_args()

    input_files = glob(f"../output/{args.source_data}/{args.model_group}/*.csv")

    for f in input_files:
        df = pd.read_csv(f, dtype=object)

        df["Min_K_Responses"] = df["Min_K_Responses"].apply(safe_literal_eval)

        # Verify the conversion
        assert isinstance(df["Min_K_Responses"][0], dict)
                                        
        df['rouge_sim_scores'] = df['rouge_sim_scores'].apply(parse_rouge_scores)
        df['cos_sim_scores'] = df['cos_sim_scores'].apply(float)


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


        df = df.drop(columns=['rouge_sim_scores', 'Min_K_Responses'])

        result_filename = f'results_{(f.split("/")[-1])}'
        result_dir = f'{args.source_data}/{args.model_group}'
        os.makedirs(result_dir, exist_ok=True)
        file_path = os.path.join(result_dir, result_filename)
        df.to_csv(file_path)
        print(f"Parsed {f}")
        


>>>>>>> origin/jessica-dev
