"""
T5 trainer for book question answering.
"""

import os
import torch
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict
from typing import Dict, Any, Optional
import logging
from .data_preprocessor import BookDataPreprocessor
from .model_config import T5BookQAConfig, T5BookQAModel
from .evaluator import T5BookQAEvaluator

logger = logging.getLogger(__name__)

class T5BookQATrainer:
    """Main trainer class for T5 book question answering."""
    
    def __init__(self, 
                 config: Optional[T5BookQAConfig] = None,
                 output_dir: str = "./results",
                 data_dir: str = "./data"):
        
        self.config = config or T5BookQAConfig()
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Initialize components
        self.preprocessor = BookDataPreprocessor()
        self.model_wrapper = T5BookQAModel(self.config)
        self.evaluator = T5BookQAEvaluator()
        
        # Data and model
        self.dataset = None
        self.trainer = None
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    
    def prepare_data(self, num_samples: Optional[int] = None, use_subset: bool = True):
        """Prepare the dataset for training."""
        logger.info("Preparing data...")
        
        # Load raw dataset
        raw_dataset = self.preprocessor.load_book_dataset()
        
        # Process into T5 format
        if use_subset and num_samples is None:
            num_samples = 50  # Small subset for initial testing
            
        self.dataset = self.preprocessor.process_dataset(raw_dataset, num_samples)
        
        # Tokenize the dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = self.dataset.map(
            self.model_wrapper.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        self.dataset = tokenized_dataset
        
        # Save processed data
        data_path = os.path.join(self.data_dir, "processed_dataset")
        self.dataset.save_to_disk(data_path)
        logger.info(f"Processed dataset saved to {data_path}")
        
        # Show sample
        self._show_data_sample()
    
    def _show_data_sample(self):
        """Show a sample of the processed data."""
        if self.dataset is None:
            return
            
        logger.info("Sample data:")
        train_sample = self.dataset["train"][0]
        
        # Decode the tokenized data to show what it looks like
        input_text = self.model_wrapper.tokenizer.decode(train_sample["input_ids"], skip_special_tokens=True)
        target_text = self.model_wrapper.tokenizer.decode(train_sample["labels"], skip_special_tokens=True)
        
        logger.info(f"Input: {input_text}")
        logger.info(f"Target: {target_text}")
    
    def setup_trainer(self):
        """Setup the Hugging Face trainer."""
        if self.dataset is None:
            raise ValueError("Dataset must be prepared first")
        
        # Load model and tokenizer
        self.model_wrapper.load_model_and_tokenizer()
        
        # Get training arguments
        training_args = self.config.get_training_args(self.output_dir)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.model_wrapper.tokenizer,
            compute_metrics=self.evaluator.compute_metrics
        )
        
        logger.info("Trainer setup complete")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Train the model."""
        if self.trainer is None:
            self.setup_trainer()
        
        logger.info("Starting training...")
        
        # Train the model
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        self.trainer.save_model(final_model_path)
        self.model_wrapper.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Training complete. Model saved to {final_model_path}")
    
    def evaluate(self):
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Model must be trained first")
        
        logger.info("Evaluating model...")
        
        # Evaluate on validation set
        eval_results = self.trainer.evaluate()
        
        # Generate predictions for detailed evaluation
        predictions = self.trainer.predict(self.dataset["validation"])
        
        # Decode predictions
        decoded_preds = self.model_wrapper.tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        decoded_labels = self.model_wrapper.tokenizer.batch_decode(
            predictions.label_ids, skip_special_tokens=True
        )
        
        # Compute additional metrics
        detailed_results = self.evaluator.evaluate_predictions(
            decoded_preds, decoded_labels
        )
        
        # Combine results
        eval_results.update(detailed_results)
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")
        
        return eval_results
    
    def quick_test(self, questions_and_contexts: list):
        """Quick test of the model with sample questions."""
        if self.model_wrapper.model is None:
            raise ValueError("Model must be loaded first")
        
        logger.info("Running quick test...")
        
        for i, (question, context) in enumerate(questions_and_contexts):
            answer = self.model_wrapper.generate_answer(question, context)
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Question: {question}")
            logger.info(f"Context: {context}")
            logger.info(f"Generated Answer: {answer}")
    
    def train_subset_first(self):
        """Train on a small subset first for sanity check."""
        logger.info("Training on subset first (sanity check)...")
        
        # Prepare small subset
        self.prepare_data(num_samples=10, use_subset=True)
        
        # Setup trainer with reduced epochs
        original_epochs = self.config.num_epochs
        self.config.num_epochs = 1
        self.setup_trainer()
        
        # Train
        self.train()
        
        # Quick evaluation
        sample_qa = [
            ("Who wrote The Great Gatsby?", "The Great Gatsby is a novel by F. Scott Fitzgerald."),
            ("When was 1984 published?", "1984 is a novel by George Orwell published in 1949.")
        ]
        self.quick_test(sample_qa)
        
        # Restore original config
        self.config.num_epochs = original_epochs
        
        logger.info("Subset training complete. Ready for full training.")
    
    def full_training_pipeline(self, subset_first: bool = True):
        """Complete training pipeline."""
        try:
            if subset_first:
                self.train_subset_first()
                
                # Ask user if they want to continue
                logger.info("Subset training successful. Proceeding with full training...")
            
            # Full training
            logger.info("Starting full training pipeline...")
            self.prepare_data(use_subset=False)
            self.setup_trainer()
            self.train()
            
            # Evaluate
            results = self.evaluate()
            
            # Final test
            sample_qa = [
                ("Who wrote The Great Gatsby?", "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925."),
                ("What type of book is To Kill a Mockingbird?", "To Kill a Mockingbird is a novel by Harper Lee."),
                ("When was 1984 published?", "1984 was published in 1949 by George Orwell.")
            ]
            self.quick_test(sample_qa)
            
            logger.info("Full training pipeline complete!")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train T5 for book question answering")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--model_name", default="t5-small", help="Model name")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--subset_first", action="store_true", help="Train on subset first")
    
    args = parser.parse_args()
    
    # Create config
    config = T5BookQAConfig(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Create trainer
    trainer = T5BookQATrainer(
        config=config,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
    
    # Run training pipeline
    trainer.full_training_pipeline(subset_first=args.subset_first)


if __name__ == "__main__":
    main()