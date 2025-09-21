"""
Evaluation metrics for T5 book question answering.
"""

import evaluate
import numpy as np
from typing import List, Dict, Any
import logging
from collections import Counter
import re
import string

logger = logging.getLogger(__name__)

class T5BookQAEvaluator:
    """Evaluator for T5 book question answering model."""
    
    def __init__(self):
        # Load evaluation metrics
        try:
            self.rouge = evaluate.load("rouge")
            self.bleu = evaluate.load("bleu")
        except Exception as e:
            logger.warning(f"Could not load some metrics: {e}")
            self.rouge = None
            self.bleu = None
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer text for evaluation."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match score."""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if self.normalize_answer(pred) == self.normalize_answer(ref):
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0
    
    def f1_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute F1 score."""
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.normalize_answer(pred).split()
            ref_tokens = self.normalize_answer(ref).split()
            
            if not ref_tokens:
                f1_scores.append(1.0 if not pred_tokens else 0.0)
                continue
            
            if not pred_tokens:
                f1_scores.append(0.0)
                continue
            
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)
            
            # Calculate overlap
            overlap = pred_counter & ref_counter
            overlap_count = sum(overlap.values())
            
            # Calculate precision and recall
            precision = overlap_count / len(pred_tokens)
            recall = overlap_count / len(ref_tokens)
            
            # Calculate F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if self.rouge is None:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            results = self.rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            
            return {
                "rouge1": results["rouge1"],
                "rouge2": results["rouge2"], 
                "rougeL": results["rougeL"]
            }
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score."""
        if self.bleu is None:
            return 0.0
        
        try:
            # BLEU expects references as list of lists
            references_formatted = [[ref.split()] for ref in references]
            predictions_formatted = [pred.split() for pred in predictions]
            
            result = self.bleu.compute(
                predictions=predictions_formatted,
                references=references_formatted
            )
            
            return result["bleu"]
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return 0.0
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate predictions against references with multiple metrics."""
        logger.info(f"Evaluating {len(predictions)} predictions...")
        
        # Core metrics
        exact_match = self.exact_match_score(predictions, references)
        f1 = self.f1_score(predictions, references)
        
        # ROUGE scores
        rouge_scores = self.rouge_scores(predictions, references)
        
        # BLEU score
        bleu = self.bleu_score(predictions, references)
        
        results = {
            "exact_match": exact_match,
            "f1": f1,
            "bleu": bleu,
            **rouge_scores
        }
        
        return results
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for Hugging Face trainer."""
        predictions, labels = eval_pred
        
        # Handle the case where predictions are logits
        if len(predictions.shape) > 2:
            predictions = np.argmax(predictions, axis=-1)
        
        # Decode predictions and labels
        # Note: This is a simplified version. In practice, you'd need access to the tokenizer
        # This method is typically called with already decoded strings
        
        if isinstance(predictions[0], str):
            # Already decoded
            decoded_preds = predictions
            decoded_labels = labels
        else:
            # Need to decode (placeholder - would need tokenizer access)
            decoded_preds = [str(pred) for pred in predictions]
            decoded_labels = [str(label) for label in labels]
        
        # Compute metrics
        results = self.evaluate_predictions(decoded_preds, decoded_labels)
        
        return results
    
    def detailed_analysis(self, predictions: List[str], references: List[str], 
                         questions: List[str] = None) -> Dict[str, Any]:
        """Perform detailed analysis of predictions."""
        results = self.evaluate_predictions(predictions, references)
        
        # Length analysis
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        results["analysis"] = {
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
            "num_samples": len(predictions)
        }
        
        # Error analysis
        errors = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if self.normalize_answer(pred) != self.normalize_answer(ref):
                error_info = {
                    "index": i,
                    "prediction": pred,
                    "reference": ref,
                    "question": questions[i] if questions and i < len(questions) else "N/A"
                }
                errors.append(error_info)
        
        results["errors"] = errors[:10]  # Top 10 errors
        results["error_rate"] = len(errors) / len(predictions)
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        print(f"Exact Match: {results['exact_match']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"BLEU Score: {results['bleu']:.4f}")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        
        if "analysis" in results:
            analysis = results["analysis"]
            print(f"\nAnalysis:")
            print(f"Average prediction length: {analysis['avg_pred_length']:.2f} words")
            print(f"Average reference length: {analysis['avg_ref_length']:.2f} words")
            print(f"Length ratio: {analysis['length_ratio']:.2f}")
            print(f"Total samples: {analysis['num_samples']}")
            print(f"Error rate: {results['error_rate']:.4f}")
        
        if "errors" in results and results["errors"]:
            print(f"\nSample Errors:")
            for i, error in enumerate(results["errors"][:3]):  # Show top 3 errors
                print(f"\nError {i+1}:")
                print(f"Question: {error['question']}")
                print(f"Predicted: {error['prediction']}")
                print(f"Reference: {error['reference']}")
        
        print("="*50)


def evaluate_model_outputs(predictions_file: str, references_file: str):
    """Evaluate model outputs from files."""
    evaluator = T5BookQAEvaluator()
    
    # Load predictions and references
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r') as f:
        references = [line.strip() for line in f]
    
    # Evaluate
    results = evaluator.detailed_analysis(predictions, references)
    evaluator.print_evaluation_report(results)
    
    return results


if __name__ == "__main__":
    # Example usage
    evaluator = T5BookQAEvaluator()
    
    # Sample predictions and references
    predictions = [
        "F. Scott Fitzgerald",
        "1949",
        "novel"
    ]
    
    references = [
        "F. Scott Fitzgerald", 
        "1949",
        "dystopian novel"
    ]
    
    results = evaluator.evaluate_predictions(predictions, references)
    evaluator.print_evaluation_report(results)