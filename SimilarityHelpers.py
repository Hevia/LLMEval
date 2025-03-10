from rouge_score import rouge_scorer
from sentence_transformers.util import cos_sim
import torch
from evaluate import load

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

def get_levenshtein_distance(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1

    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def get_bert_score_batch(gold: list[str], test: list[str]) -> float:
    bert_score = load("bertscore")
    score = bert_score.compute(predictions=test, references=gold)
    return score
