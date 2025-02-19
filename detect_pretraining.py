import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path
from tqdm import tqdm

class PretrainingDetector:
    def __init__(self, model_name):
        """Initialize the detector with a HuggingFace model."""
        self.model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, device_map='auto')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def calculate_token_probabilities(self, text: str) -> list:
        """Calculate log probabilities for each token in the text."""
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits
            
        # Get log probabilities for each token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        all_probs = []
        
        # Extract probability for each actual token
        input_ids_processed = input_ids[0][1:]  # Skip first token for causal LM
        for i, token_id in enumerate(input_ids_processed):
            probability = log_probs[0, i, token_id].item()
            all_probs.append(probability)
            
        return all_probs

    def get_min_k_probability(self, text: str, k_ratio: float) -> float:
        """
        Calculate the Min-K% probability score for the text.
        
        Args:
            text: Input text to analyze
            k_ratio: Ratio of tokens to consider (e.g., 0.1 for 10%)
            
        Returns:
            Average log probability of the k% least likely tokens
        """
        # Get token probabilities
        token_probs = self.calculate_token_probabilities(text)
        
        # Calculate k lowest probabilities
        k_length = int(len(token_probs) * k_ratio)
        lowest_k_probs = np.sort(token_probs)[:k_length]
        
        # Return negative mean (higher score = more likely to be in training data)
        return -np.mean(lowest_k_probs).item()

def detect_pretraining(text: str, model_name: str, k_ratios=[0.05, 0.1, 0.2, 0.3]) -> dict:
    """
    Detect if text was likely in the model's training data using Min-K% probability.
    
    Args:
        text: Text to analyze
        model_name: HuggingFace model name
        k_ratios: List of K% ratios to test
        
    Returns:
        Dictionary of Min-K% probability scores for each k_ratio
    """
    detector = PretrainingDetector(model_name)
    results = {}
    
    for ratio in k_ratios:
        score = detector.get_min_k_probability(text, ratio)
        results[f"Min_{ratio*100}% Prob"] = score
        
    return results

# Example usage:
# if __name__ == "__main__":
#     text = "Sample text to analyze for pretraining detection"
#     model_name = "gpt2"  # or any other HuggingFace model

#    results = detect_pretraining(text, model_name)
#    print("Detection Results:")
#    for k, score in results.items():
#        print(f"{k}: {score:.4f}") 