import pandas as pd
import argparse
from glob import glob
import pprint
import prepare_clean

def required(result_type):
    if result_type == "scores":
        return {
            'Model', 'Task_Prefix', 'Dataset_Name', 'Model_Responses',
            'Gold_Labels', 'cos_sim_scores', 'levenshtein_distance',
            'rouge1_precision','rouge1_recall', 'rouge1_f1', 'rouge2_precision', 
            'rouge2_recall','rouge2_f1', 'rougeL_precision', 'rougeL_recall', 
            'rougeL_f1', 'Min_10.0%_Prob', 'Min_20.0%_Prob', 'Min_30.0%_Prob', 
            'Min_40.0%_Prob', 'Min_50.0%_Prob'
        }
    else: # classifiers
        return {
            'Model', 'Task_Prefix', 'Dataset_Name', 'Model_Responses',
            'Gold_Labels', 'GradientBoosting_prediction_x', 'GradientBoosting_probability_x',
            'GradientBoosting_status', 'LogisticRegression_prediction_x',
            'LogisticRegression_probability_x', 'LogisticRegression_status',
            'RandomForest_prediction_x', 'RandomForest_probability_x',
            'RandomForest_status', 'GradientBoosting_prediction_y',
            'GradientBoosting_probability_y', 'GradientBoosting_label',
            'LogisticRegression_prediction_y', 'LogisticRegression_probability_y',
            'LogisticRegression_label', 'RandomForest_prediction_y',
            'RandomForest_probability_y', 'RandomForest_label'
        }

def check_missing(result_type, source, group):
    # generate required columns
    required_columns = required(result_type)

    # generate relevant file names
    files = glob(f"./{result_type}/{source}/{group}/*.csv")

    # store issues
    issues = dict()

    # loop over files
    for f in files:
        name = f.split("/")[-1]
        df = pd.read_csv(f, dtype=object)

        # search for missing stuff
        missing_columns = required_columns - set(df.columns)
        columns_with_missing = df.columns[df.isnull().any()].tolist()

        if missing_columns: # first check for columns that are missing
            issues[name] = {"missing_columns": missing_columns}
        
        if len(columns_with_missing): # then check for missing values in existing columns
            issues.setdefault(name, {}).update({"has_missing_values": columns_with_missing})
    
    if issues:
        print(f"{result_type}/{source}/{group} data is not valid:")
        pprint.pprint(issues)
    else:
        print("All files have required columns and no missing values.")
    return

def run(source, group):
    check_missing("scores", source, group)
    check_missing("classifiers", source, group)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_data", type=str)
    parser.add_argument("model_group", type=str)
    
    args= parser.parse_args()

    run(args.source_data, args.model_group)


