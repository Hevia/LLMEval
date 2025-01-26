from rouge_score import rouge_scorer
from sentence_transformers.util import cos_sim
import torch

def get_cos_similarity_score(gold_embedding: torch.Tensor, test_embedding: torch.Tensor) -> float:
    """
    Gets the similarity score between the gold and test strings using the cosine similarity score from the sentence-transformers library.
    """
    sim = cos_sim(gold_embedding, test_embedding)
    return sim.item()

def get_rouge_scores(gold: str, test: str) -> dict:
    """
    Gets the ROUGE score between the gold and test strings using the rouge_score library.
    Returns a dictionary with the ROUGE scores for each metric.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(gold, test)
    return score

