"""
T5 model configuration for book question answering.
"""

from transformers import (
    T5Config, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    TrainingArguments
)
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class T5BookQAConfig:
    """Configuration class for T5 book question answering model."""
    
    def __init__(self, 
                 model_name: str = "t5-small",
                 max_input_length: int = 512,
                 max_output_length: int = 128,
                 train_batch_size: int = 8,
                 eval_batch_size: int = 8,
                 learning_rate: float = 3e-4,
                 num_epochs: int = 3,
                 warmup_steps: int = 500,
                 weight_decay: float = 0.01,
                 save_steps: int = 1000,
                 eval_steps: int = 500,
                 logging_steps: int = 100):
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
    def get_training_args(self, output_dir: str) -> TrainingArguments:
        """Get training arguments for the model."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=[],  # Disable wandb/tensorboard for simplicity
        )


class T5BookQAModel:
    """Wrapper class for T5 model and tokenizer."""
    
    def __init__(self, config: T5BookQAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model_and_tokenizer(self):
        """Load the pre-trained T5 model and tokenizer."""
        logger.info(f"Loading {self.config.model_name} model and tokenizer...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
        
    def preprocess_function(self, examples):
        """Preprocess examples for T5 training."""
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt" if isinstance(inputs, str) else None
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_output_length,
                truncation=True,
                padding=True,
                return_tensors="pt" if isinstance(targets, str) else None
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a given question and context."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Format input
        input_text = f"question: {question} context: {context}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_output_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        
        logger.info("Fine-tuned model loaded successfully")


def get_model_info():
    """Get information about available T5 models."""
    models = {
        "t5-small": {
            "parameters": "60M",
            "description": "Smallest T5 model, good for quick experimentation",
            "recommended_batch_size": 8
        },
        "t5-base": {
            "parameters": "220M", 
            "description": "Base T5 model, balanced performance",
            "recommended_batch_size": 4
        },
        "t5-large": {
            "parameters": "770M",
            "description": "Large T5 model, better performance but slower",
            "recommended_batch_size": 2
        }
    }
    
    return models


if __name__ == "__main__":
    # Example usage
    config = T5BookQAConfig()
    model = T5BookQAModel(config)
    model.load_model_and_tokenizer()
    
    # Test generation
    question = "Who wrote The Great Gatsby?"
    context = "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925."
    answer = model.generate_answer(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")