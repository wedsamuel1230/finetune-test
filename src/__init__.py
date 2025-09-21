"""
T5 Fine-tuning for Book Question Answering

A complete pipeline for fine-tuning T5-small model on book datasets
for domain-specific question answering.
"""

from .data_preprocessor import BookDataPreprocessor
from .model_config import T5BookQAConfig, T5BookQAModel
from .trainer import T5BookQATrainer
from .evaluator import T5BookQAEvaluator

__version__ = "1.0.0"
__author__ = "T5 Book QA Team"

__all__ = [
    "BookDataPreprocessor",
    "T5BookQAConfig", 
    "T5BookQAModel",
    "T5BookQATrainer",
    "T5BookQAEvaluator"
]