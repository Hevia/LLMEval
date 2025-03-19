import joblib
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('..'))
from train_classifiers import *
import prepare_clean

if __name__ == "__main__":
    metadata = [
        "Model",
        "Task_Prefix",
        "Dataset_Name",
        "Model_Responses",
        "Gold_Labels",
        "Min_K_Responses",
        "cos_sim_scores",
        "rouge_sim_scores",
        "levenshtein_distance"
    ]

    targets = [
        "GradientBoosting",
        "LogisticRegression"
    ]

    # datasets = ["CNN-DailyMail", "Samsum", "Xsum"]
    # models = glob.glob("../output/models/*.joblib")

    # Process CSV files and extract data
    print("Processing CSV files...")
    df: pd.DataFrame = process_csv_files("../output")
    
    # Validate features before extraction
    print("\nValidating features...")
    validated_df = validate_features(df)
    print(f"Rows after validation: {len(validated_df)} (original: {len(df)})")

    # Extract features
    print("\nExtracting features...")
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    X, y, feature_names = extract_features(validated_df)

    # Cleaned data
    clean = validated_df[metadata]

    # # loop over files
    # for dataset in datasets:
    #     data = pd.read_csv(f"./validated/{dataset}_validated.csv")
    #     X, y, feature_names = extract_features(data)

        # loop over models
    for model in targets:
        # clf = joblib.load(model)
        clf = joblib.load(f"../output/models/{model}.joblib")
        print(f"Predicting {model} results...")
        y_pred = clf.predict(X)
        
        clean[f"{model}_prediction"] = y_pred.tolist()

        # write to output

    for dataset, g in clean.groupby("Dataset_Name"):
        g.to_csv(f"./validated/{dataset}_prediction.csv")
        print(f"{dataset} predictions complete...")
    
    print("preparing for data analysis...")
    prepare_clean.process()
    print("DONE")
        


