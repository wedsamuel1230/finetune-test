#!/usr/bin/env python3
"""
Command-line training script for T5 Book QA model.
"""

import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trainer import T5BookQATrainer
from model_config import T5BookQAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train T5 for book question answering")
    
    # Model arguments
    parser.add_argument("--model_name", default="t5-small", 
                       choices=["t5-small", "t5-base", "t5-large"],
                       help="Pre-trained T5 model to use")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--max_input_length", type=int, default=512,
                       help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=128,
                       help="Maximum output sequence length")
    
    # Data arguments
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use (None for all)")
    parser.add_argument("--subset_first", action="store_true",
                       help="Train on small subset first for sanity check")
    
    # I/O arguments
    parser.add_argument("--output_dir", default="./results",
                       help="Output directory for model and logs")
    parser.add_argument("--data_dir", default="./data",
                       help="Data directory for processed datasets")
    
    # Workflow arguments
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training and only evaluate")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only run evaluation")
    
    args = parser.parse_args()
    
    logger.info("🚀 T5 Book QA Training")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output dir: {args.output_dir}")
    
    try:
        # Create configuration
        config = T5BookQAConfig(
            model_name=args.model_name,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        
        # Create trainer
        trainer = T5BookQATrainer(
            config=config,
            output_dir=args.output_dir,
            data_dir=args.data_dir
        )
        
        if args.evaluate_only:
            logger.info("Running evaluation only...")
            
            # Load existing model
            model_path = os.path.join(args.output_dir, "final_model")
            if os.path.exists(model_path):
                trainer.model_wrapper.load_finetuned_model(model_path)
                results = trainer.evaluate()
                logger.info("Evaluation complete!")
            else:
                logger.error(f"No trained model found at {model_path}")
                return 1
        
        elif args.skip_training:
            logger.info("Skipping training, preparing data only...")
            trainer.prepare_data(num_samples=args.num_samples, use_subset=False)
            logger.info("Data preparation complete!")
        
        else:
            # Full training pipeline
            if args.subset_first:
                logger.info("Running subset training first...")
                trainer.train_subset_first()
                
                # Ask user if they want to continue
                response = input("\nSubset training complete. Continue with full training? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    logger.info("Training stopped by user.")
                    return 0
            
            # Run full training
            logger.info("Starting full training pipeline...")
            results = trainer.full_training_pipeline(subset_first=False)
            
            logger.info("🎉 Training complete!")
            logger.info(f"Results: {results}")
            
            # Final test
            logger.info("\n🧪 Running final test...")
            sample_qa = [
                ("Who wrote The Great Gatsby?", "The Great Gatsby is a novel by F. Scott Fitzgerald."),
                ("When was 1984 published?", "1984 was published in 1949 by George Orwell."),
                ("What type of book is To Kill a Mockingbird?", "To Kill a Mockingbird is a novel by Harper Lee.")
            ]
            trainer.quick_test(sample_qa)
        
        logger.info("✅ All operations completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())