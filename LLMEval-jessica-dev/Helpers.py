import csv
import pandas as pd
import numpy as np
import prompting.prompts as pm
import spacy
from tqdm import tqdm
from Retriever import PaperNode
import json
from datetime import datetime
import traceback

def log_debug(file_name, message):
    """Helper function to log debug messages with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    with open(file_name, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_error(file_name, error, include_traceback=True):
    """
    Helper function to log error messages with timestamps and optional traceback
    
    Args:
        file_name: Path to the log file
        error: The error object or error message string
        include_traceback: Whether to include the full traceback (default: True)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    with open(file_name, "a") as f:
        if isinstance(error, Exception):
            f.write(f"[{timestamp}] ERROR: {error.__class__.__name__}: {str(error)}\n")
            if include_traceback:
                f.write(f"{traceback.format_exc()}\n")
        else:
            f.write(f"[{timestamp}] ERROR: {str(error)}\n")

def get_question(domain_no, question_no):
    return pm.signaling_questions[domain_no - 1][question_no - 1]
