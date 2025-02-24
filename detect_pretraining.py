import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

class PretrainingDetector:
    def __init__(self, model_name, epsilon=0.5):
        """
        Initialize detector with model and decision threshold.
        
        Args:
            model_name: Name of HuggingFace model
            epsilon: Decision threshold for membership
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True, device_map='auto')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.epsilon = epsilon

    def calculate_token_probabilities(self, text: str) -> list:
        """
        Calculate log p(xi|x1,...,xi-1) for each token in sequence.
        Returns list of log probabilities.
        """
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits
            
        # Get log probabilities for each token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = []
        
        # Extract log prob for each actual token given previous tokens
        input_ids_processed = input_ids[0][1:]  # Skip first token
        for i, token_id in enumerate(input_ids_processed):
            log_prob = log_probs[0, i, token_id].item()
            token_log_probs.append(log_prob)
            
        return token_log_probs

    def min_k_prob(self, text: str, k_ratio: float) -> float:
        """
        Implements Algorithm 1: Pretraining Data Detection using Min-K% Prob
        
        Args:
            text: Input sequence
            k_ratio: Percentage of lowest probability tokens to consider
            
        Returns:
            Min-K% probability score
        """
        # Get log p(xi|x1,...,xi-1) for all tokens
        token_log_probs = self.calculate_token_probabilities(text)
        
        # Select k% tokens with lowest probability
        k_size = max(1, int(len(token_log_probs) * k_ratio))
        min_k_probs = np.sort(token_log_probs)[:k_size]
        
        # Calculate average log probability of min-k tokens
        score = -np.mean(min_k_probs).item()
        
        return score
    
    def predict(self, prompt: str, gold_text: str, k_ratio: float) -> tuple[bool, float]:
        """
        Predict if prompt was derived from gold_text using Min-K% Prob.
        
        Args:
            prompt: Text to evaluate
            gold_text: Gold standard text to compare against
            k_ratio: Percentage of tokens to consider
            
        Returns:
            (is_member, score) tuple where:
            - is_member: True if prompt likely derived from gold_text
            - score: Min-K% probability score
        """
        # Get probabilities for both texts
        prompt_probs = self.calculate_token_probabilities(prompt)
        gold_probs = self.calculate_token_probabilities(gold_text)
        
        # Select k% tokens with lowest probability from prompt
        k_size = max(1, int(len(prompt_probs) * k_ratio))
        min_k_indices = np.argsort(prompt_probs)[:k_size]
        
        # Calculate score using corresponding positions in gold text
        min_k_gold_probs = [gold_probs[i] for i in min_k_indices]
        score = -np.mean(min_k_gold_probs).item()
        
        # Compare against threshold
        is_member = score <= self.epsilon
        return is_member, score

def detect_pretraining_batch(prompts: list[str], gold_texts: list[str], 
                           model_name: str, k_ratios=[0.05, 0.1, 0.2, 0.3],
                           epsilon=0.5) -> dict:
    """
    Run detection on a batch of texts comparing against gold standards.
    
    Args:
        prompts: List of texts to evaluate
        gold_texts: List of gold standard texts to compare against
        model_name: HuggingFace model name
        k_ratios: List of k% values to test
        epsilon: Decision threshold
    """
    detector = PretrainingDetector(model_name, epsilon)
    results = {}
    
    for ratio in k_ratios:
        scores = []
        predictions = []
        for prompt, gold in zip(prompts, gold_texts):
            is_member, score = detector.predict(prompt, gold, ratio)
            scores.append(score)
            predictions.append(int(is_member))
            
        results[f"Min_{ratio*100}% Prob"] = {
            'scores': scores,
            'predictions': predictions,
            'gold_texts': gold_texts,
            'prompts': prompts
        }
        
    return results