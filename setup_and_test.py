#!/usr/bin/env python3
"""
Quick setup and test script for T5 Book QA project.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        logger.info(f"Success: {description}")
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_dependencies():
    """Check if required packages are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies satisfied!")
    return True

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    logger.info("Testing data preprocessing...")
    
    try:
        sys.path.insert(0, 'src')
        from data_preprocessor import BookDataPreprocessor
        
        preprocessor = BookDataPreprocessor()
        dataset = preprocessor._create_sample_dataset()
        processed = preprocessor.process_dataset(dataset, num_samples=5)
        
        logger.info(f"✓ Generated {len(processed['train'])} training samples")
        logger.info(f"✓ Generated {len(processed['validation'])} validation samples")
        
        # Show sample
        sample = processed['train'][0]
        logger.info(f"Sample input: {sample['input_text'][:100]}...")
        logger.info(f"Sample target: {sample['target_text']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing test failed: {e}")
        return False

def test_model_config():
    """Test model configuration."""
    logger.info("Testing model configuration...")
    
    try:
        sys.path.insert(0, 'src')
        from model_config import T5BookQAConfig, get_model_info
        
        config = T5BookQAConfig()
        logger.info(f"✓ Model: {config.model_name}")
        logger.info(f"✓ Max input length: {config.max_input_length}")
        logger.info(f"✓ Training epochs: {config.num_epochs}")
        
        model_info = get_model_info()
        logger.info(f"✓ Available models: {list(model_info.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model config test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality."""
    logger.info("Testing evaluation...")
    
    try:
        sys.path.insert(0, 'src')
        from evaluator import T5BookQAEvaluator
        
        evaluator = T5BookQAEvaluator()
        
        # Test with sample data
        predictions = ["F. Scott Fitzgerald", "1925", "novel"]
        references = ["F. Scott Fitzgerald", "1925", "book"]
        
        results = evaluator.evaluate_predictions(predictions, references)
        
        logger.info(f"✓ Exact Match: {results['exact_match']:.3f}")
        logger.info(f"✓ F1 Score: {results['f1']:.3f}")
        logger.info(f"✓ BLEU Score: {results['bleu']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation test failed: {e}")
        return False

def run_example_demos():
    """Run example demonstrations."""
    logger.info("Running example demos...")
    
    try:
        # Test hybrid routing
        logger.info("Testing hybrid routing...")
        result = subprocess.run([sys.executable, "examples/hybrid_routing.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("✓ Hybrid routing demo completed")
        else:
            logger.warning(f"Hybrid routing demo issues: {result.stderr}")
        
        # Test inference demo  
        logger.info("Testing inference demo...")
        result = subprocess.run([sys.executable, "examples/inference_demo.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("✓ Inference demo completed")
        else:
            logger.warning(f"Inference demo issues: {result.stderr}")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo test failed: {e}")
        return False

def run_unit_tests():
    """Run unit tests."""
    logger.info("Running unit tests...")
    
    try:
        result = subprocess.run([sys.executable, "tests/test_pipeline.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✓ All unit tests passed")
            return True
        else:
            logger.warning(f"Some unit tests failed:\n{result.stdout}")
            return False
            
    except Exception as e:
        logger.error(f"Unit test execution failed: {e}")
        return False

def create_sample_files():
    """Create sample input/output files for testing."""
    logger.info("Creating sample files...")
    
    try:
        # Create sample questions file
        sample_questions = [
            {
                "question": "Who wrote The Great Gatsby?",
                "context": "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925.",
                "expected": "F. Scott Fitzgerald"
            },
            {
                "question": "When was 1984 published?", 
                "context": "1984 is a dystopian novel by George Orwell published in 1949.",
                "expected": "1949"
            }
        ]
        
        import json
        with open('sample_questions.json', 'w') as f:
            json.dump(sample_questions, f, indent=2)
        
        logger.info("✓ Created sample_questions.json")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample files: {e}")
        return False

def main():
    """Main setup and test function."""
    logger.info("🚀 T5 Book QA Setup and Test")
    logger.info("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Configuration", test_model_config),
        ("Evaluation", test_evaluation),
        ("Example Demos", run_example_demos),
        ("Unit Tests", run_unit_tests),
        ("Sample Files", create_sample_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("-" * 30)
        results[test_name] = test_func()
    
    # Summary
    logger.info(f"\n📊 SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! The project is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Open notebooks/t5_book_qa_training.ipynb in Google Colab")
        logger.info("2. Run the training pipeline")
        logger.info("3. Test with your own book data")
    else:
        logger.info(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)