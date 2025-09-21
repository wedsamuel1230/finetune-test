#!/usr/bin/env python3
"""
Quick demo script to test the T5 Book QA model.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_data_processing():
    """Demo the data processing pipeline."""
    print("🔧 Data Processing Demo")
    print("=" * 40)
    
    try:
        from data_preprocessor import BookDataPreprocessor
        
        preprocessor = BookDataPreprocessor()
        
        # Load sample dataset
        print("Loading book dataset...")
        dataset = preprocessor.load_book_dataset()
        
        # Process into T5 format
        print("Processing into T5 format...")
        processed = preprocessor.process_dataset(dataset, num_samples=5)
        
        # Show samples
        print(f"\nGenerated {len(processed['train'])} training samples")
        print("\nSample Q&A pair:")
        sample = processed['train'][0]
        print(f"Input: {sample['input_text']}")
        print(f"Target: {sample['target_text']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_model_config():
    """Demo the model configuration."""
    print("\n🤖 Model Configuration Demo")
    print("=" * 40)
    
    try:
        from model_config import T5BookQAConfig, get_model_info
        
        config = T5BookQAConfig()
        print(f"Model: {config.model_name}")
        print(f"Max input length: {config.max_input_length}")
        print(f"Max output length: {config.max_output_length}")
        print(f"Training epochs: {config.num_epochs}")
        print(f"Batch size: {config.train_batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        
        print("\nAvailable models:")
        models = get_model_info()
        for name, info in models.items():
            print(f"- {name}: {info['parameters']} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_evaluation():
    """Demo the evaluation functionality."""
    print("\n📊 Evaluation Demo")
    print("=" * 40)
    
    try:
        from evaluator import T5BookQAEvaluator
        
        evaluator = T5BookQAEvaluator()
        
        # Sample predictions and references
        predictions = [
            "F. Scott Fitzgerald",
            "1949", 
            "dystopian novel"
        ]
        
        references = [
            "F. Scott Fitzgerald",
            "1949",
            "dystopian social science fiction novel"
        ]
        
        print("Sample predictions vs references:")
        for pred, ref in zip(predictions, references):
            print(f"Predicted: {pred}")
            print(f"Reference: {ref}")
            print()
        
        # Evaluate
        results = evaluator.evaluate_predictions(predictions, references)
        
        print("Evaluation Results:")
        print(f"Exact Match: {results['exact_match']:.3f}")
        print(f"F1 Score: {results['f1']:.3f}")
        print(f"BLEU Score: {results['bleu']:.3f}")
        print(f"ROUGE-1: {results['rouge1']:.3f}")
        print(f"ROUGE-L: {results['rougeL']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_hybrid_routing():
    """Demo the hybrid routing system."""
    print("\n🔄 Hybrid Routing Demo")
    print("=" * 40)
    
    try:
        from examples.hybrid_routing import HybridQARouter
        
        router = HybridQARouter()
        
        test_questions = [
            "Who wrote The Great Gatsby?",
            "What is the capital of France?",
            "What type of book is 1984?", 
            "How does photosynthesis work?"
        ]
        
        print("Routing decisions:")
        for question in test_questions:
            is_domain = router.is_domain_question(question)
            route = "📚 Book Domain (SLM)" if is_domain else "🌍 General (LLM)"
            print(f"'{question}' → {route}")
        
        # Show routing stats
        stats = router.get_routing_stats(test_questions)
        print(f"\nRouting Statistics:")
        print(f"Domain questions: {stats['domain_questions']}/{stats['total_questions']} ({stats['domain_percentage']:.1f}%)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Demo T5 Book QA components")
    parser.add_argument("--component", 
                       choices=["data", "model", "eval", "routing", "all"],
                       default="all",
                       help="Component to demo")
    
    args = parser.parse_args()
    
    print("🎯 T5 Book QA Demo")
    print("=" * 50)
    
    demos = {
        "data": demo_data_processing,
        "model": demo_model_config,
        "eval": demo_evaluation,
        "routing": demo_hybrid_routing
    }
    
    success_count = 0
    total_count = 0
    
    if args.component == "all":
        for name, demo_func in demos.items():
            total_count += 1
            if demo_func():
                success_count += 1
    else:
        total_count = 1
        if demos[args.component]():
            success_count = 1
    
    print(f"\n📊 Demo Summary")
    print("=" * 50)
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python train.py --subset_first")
        print("3. Open the Colab notebook for GPU training")
    else:
        print("⚠️  Some demos failed. Install dependencies first:")
        print("pip install -r requirements.txt")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())