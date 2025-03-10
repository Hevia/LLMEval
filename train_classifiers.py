#!/usr/bin/env python3
# train_classifiers.py - Trains classifiers to detect contaminated vs non-contaminated examples

import os
import json
import glob
import numpy as np
import pandas as pd
import re
from ast import literal_eval
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


def process_csv_files(root_dir: str = './output') -> pd.DataFrame:
    """
    Recursively process all CSV files in the given directory structure.
    Label files from Control as 0 (not contaminated) and from Treatment as 1 (contaminated).
    
    Args:
        root_dir: Root directory containing the dataset folders
        
    Returns:
        DataFrame with extracted features and labels
    """
    all_data: List[pd.DataFrame] = []
    
    # Get all dataset directories
    dataset_dirs: List[str] = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for dataset in dataset_dirs:
        dataset_path: str = os.path.join(root_dir, dataset)
        
        # Process Control directory (not contaminated, label 0)
        control_path: str = os.path.join(dataset_path, 'Control')
        if os.path.exists(control_path):
            control_files: List[str] = glob.glob(os.path.join(control_path, '*.csv'))
            for csv_file in control_files:
                print(f"Processing control file: {csv_file}")
                df = pd.read_csv(csv_file)
                df['label'] = 0  # Not contaminated
                all_data.append(df)
        
        # Process Treatment directory (contaminated, label 1)
        treatment_path: str = os.path.join(dataset_path, 'Treatment')
        if os.path.exists(treatment_path):
            treatment_files: List[str] = glob.glob(os.path.join(treatment_path, '*.csv'))
            for csv_file in treatment_files:
                print(f"Processing treatment file: {csv_file}")
                df = pd.read_csv(csv_file)
                df['label'] = 1  # Contaminated
                all_data.append(df)
    
    # Combine all data
    if not all_data:
        raise ValueError("No CSV files found in the specified directory structure")
    
    combined_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
    return combined_df


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract numerical features from DataFrame.
    Handles the JSON-like structure in Min_K_Responses and rouge_sim_scores.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    # Process and extract features
    features: List[Dict[str, float]] = []
    
    # Parse the JSON-like structures in Min_K_Responses and rouge_sim_scores
    for _, row in df.iterrows():
        row_features: Dict[str, float] = {}
        
        # Extract cos_sim_scores
        if pd.notna(row['cos_sim_scores']):
            row_features['cos_sim'] = float(row['cos_sim_scores'])
        else:
            row_features['cos_sim'] = 0.0
            
        # Extract levenshtein_distance
        if pd.notna(row['levenshtein_distance']):
            row_features['levenshtein'] = float(row['levenshtein_distance'])
        else:
            row_features['levenshtein'] = 0.0
            
        # Extract Min_K_Responses features
        if pd.notna(row['Min_K_Responses']):
            try:
                # Convert to string to ensure consistent processing
                min_k_text = str(row['Min_K_Responses'])
                
                # Define regex pattern to extract Min_K values
                # This pattern matches key-value pairs like: 'Min_10.0% Prob': 14.023460388183594
                min_k_pattern = r"'(Min_[\d.]+%\s+Prob)':\s*([\d.]+)"
                
                # Find all matches in the text
                matches = re.findall(min_k_pattern, min_k_text)
                
                if matches:
                    for k, v in matches:
                        # Clean up the key name
                        clean_key = k.replace("'", "").replace("%", "").replace(" ", "_")
                        row_features[f'min_k_{clean_key}'] = float(v)
                else:
                    # Set default values if no matches found
                    row_features['min_k_Min_10_Prob'] = 0.0
                    row_features['min_k_Min_20_Prob'] = 0.0
                    row_features['min_k_Min_30_Prob'] = 0.0
                    row_features['min_k_Min_40_Prob'] = 0.0
                    row_features['min_k_Min_50_Prob'] = 0.0
            except Exception as e:
                print(f"Error parsing Min_K_Responses [{type(e).__name__}]: {e}")
                print(f"Problematic value: {row['Min_K_Responses'][:100]}...")  # Print first 100 chars
                
                # Set default values
                row_features['min_k_Min_10_Prob'] = 0.0
                row_features['min_k_Min_20_Prob'] = 0.0
                row_features['min_k_Min_30_Prob'] = 0.0
                row_features['min_k_Min_40_Prob'] = 0.0
                row_features['min_k_Min_50_Prob'] = 0.0
        else:
            # Set default values if Min_K_Responses is NA
            row_features['min_k_Min_10_Prob'] = 0.0
            row_features['min_k_Min_20_Prob'] = 0.0
            row_features['min_k_Min_30_Prob'] = 0.0
            row_features['min_k_Min_40_Prob'] = 0.0
            row_features['min_k_Min_50_Prob'] = 0.0
        
        # Extract rouge_sim_scores features
        if pd.notna(row['rouge_sim_scores']):
            try:
                # Using regex patterns to extract rouge scores instead of string replacement
                rouge_metrics = {'rouge1': {}, 'rouge2': {}, 'rougeL': {}}
                
                # Define regex patterns to extract values
                patterns = {
                    'rouge1': r"rouge1'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)",
                    'rouge2': r"rouge2'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)",
                    'rougeL': r"rougeL'?: Score\(precision=([\d.]+), recall=([\d.]+), fmeasure=([\d.]+)\)"
                }
                
                rouge_text = str(row['rouge_sim_scores'])
                
                # Extract values using regex for each rouge type
                for rouge_type, pattern in patterns.items():
                    match = re.search(pattern, rouge_text)
                    if match:
                        precision, recall, fmeasure = match.groups()
                        row_features[f'{rouge_type}_precision'] = float(precision)
                        row_features[f'{rouge_type}_recall'] = float(recall)
                        row_features[f'{rouge_type}_fmeasure'] = float(fmeasure)
                    else:
                        # Set default values if pattern not found
                        row_features[f'{rouge_type}_precision'] = 0.0
                        row_features[f'{rouge_type}_recall'] = 0.0
                        row_features[f'{rouge_type}_fmeasure'] = 0.0
                        
            except Exception as e:
                # More generic exception handling with detailed error information
                print(f"Error parsing rouge_sim_scores [{type(e).__name__}]: {e}")
                print(f"Problematic value: {row['rouge_sim_scores'][:100]}...")  # Print first 100 chars
                
                # Set default values for rouge metrics
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    for metric in ['precision', 'recall', 'fmeasure']:
                        row_features[f'{rouge_type}_{metric}'] = 0.0
        else:
            # Set default values if rouge_sim_scores is NA
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                for metric in ['precision', 'recall', 'fmeasure']:
                    row_features[f'{rouge_type}_{metric}'] = 0.0
        
        features.append(row_features)
    
    # Convert to DataFrame
    feature_df: pd.DataFrame = pd.DataFrame(features)
    
    # Handle missing values
    feature_df.fillna(0, inplace=True)
    
    # Extract labels
    y: np.ndarray = df['label'].values
    
    # Create feature matrix
    X: np.ndarray = feature_df.values
    
    return X, y, feature_df.columns.tolist()


def train_classifiers(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Train and evaluate multiple classifiers using GridSearchCV and cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        
    Returns:
        Dictionary with trained classifiers and performance metrics
        X_test: Test features
        y_test: Test labels
    """
    # Split data into train and test sets
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define classifiers and their hyperparameters
    classifiers: Dict[str, Dict[str, Union[BaseEstimator, Dict[str, List[Any]]]]] = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        },
        'SVC': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto', 0.1, 1]
            }
        }
    }
    
    # Train and evaluate each classifier
    results: Dict[str, Dict[str, Any]] = {}
    cv: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf_info in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Define pipeline with scaling
        pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf_info['model'])
        ])
        
        # Perform grid search
        grid_search: GridSearchCV = GridSearchCV(
            pipeline,
            clf_info['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model: Pipeline = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred: np.ndarray = best_model.predict(X_test)
        y_pred_proba: np.ndarray = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred, zero_division=0)
        recall: float = recall_score(y_test, y_pred, zero_division=0)
        f1: float = f1_score(y_test, y_pred, zero_division=0)
        roc_auc: float = roc_auc_score(y_test, y_pred_proba)
        
        # Generate confusion matrix
        conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance: Optional[Dict[str, float]] = None
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = dict(zip(feature_names, 
                                          best_model.named_steps['classifier'].feature_importances_))
        elif name == 'LogisticRegression':
            feature_importance = dict(zip(feature_names, 
                                          best_model.named_steps['classifier'].coef_[0]))
        
        # Store results
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
        
        # Print results
        print(f"\n{name} Results:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Cross-Validation Score: {grid_search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        if feature_importance:
            print("\nFeature Importance:")
            sorted_features: List[Tuple[str, float]] = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, importance in sorted_features[:10]:  # Top 10 features
                print(f"{feature}: {importance:.4f}")
    
    return results, X_test, y_test


def plot_results(results: Dict[str, Dict[str, Any]], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Generate plots for model comparison and ROC curves.
    
    Args:
        results: Dictionary with classifier results
        X_test: Test features
        y_test: Test labels
    """
    # Create directory for plots if it doesn't exist
    os.makedirs("./plots", exist_ok=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [r['accuracy'] for r in results.values()])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/model_accuracy_comparison.png')
    
    # Plot F1 score comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [r['f1'] for r in results.values()])
    plt.title('Model F1 Score Comparison')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/model_f1_comparison.png')
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        model: Pipeline = result['model']
        y_pred_proba: np.ndarray = model.predict_proba(X_test)[:, 1]
        fpr: np.ndarray
        tpr: np.ndarray
        _: np.ndarray
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc: float = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/roc_curves.png')
    
    # Plot feature importance for the best model
    best_model_name: str = max(results.keys(), key=lambda k: results[k]['f1'])
    feature_importance: Optional[Dict[str, float]] = results[best_model_name]['feature_importance']
    
    if feature_importance:
        # Sort features by importance
        sorted_features: List[Tuple[str, float]] = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features: List[Tuple[str, float]] = sorted_features[:15]  # Top 15 features
        
        plt.figure(figsize=(12, 8))
        features, importances = zip(*top_features)
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('./plots/feature_importance.png')


def main() -> None:
    # Process CSV files and extract data
    print("Processing CSV files...")
    df: pd.DataFrame = process_csv_files()
    
    # Extract features
    print("\nExtracting features...")
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    X, y, feature_names = extract_features(df)
    
    print(f"\nDataset summary:")
    print(f"Total samples: {len(y)}")
    print(f"Contaminated samples: {sum(y)}")
    print(f"Non-contaminated samples: {len(y) - sum(y)}")
    print(f"Number of features: {X.shape[1]}")
    
    # Train classifiers
    print("\nTraining classifiers...")
    results: Dict[str, Dict[str, Any]]
    X_test: np.ndarray
    y_test: np.ndarray
    results, X_test, y_test = train_classifiers(X, y, feature_names)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results, X_test, y_test)
    
    print("\nTraining complete! Results saved to the plots directory.")


if __name__ == "__main__":
    main()
