import ast
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval # this will evaluate a string as literal Python objs


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



df = pd.read_csv("/Users/chenxinliu/LLMEval/output/HuggingFaceTB_SmolLM2-360M-Instruct_Samsum.csv", dtype=object)

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

df.to_csv('data_SmolLM2-360M-Instruct_Samsum.csv')
