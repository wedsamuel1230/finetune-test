"""
Basic tests for the T5 Book QA pipeline.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessor import BookDataPreprocessor
from model_config import T5BookQAConfig, get_model_info
from evaluator import T5BookQAEvaluator

class TestDataPreprocessor(unittest.TestCase):
    """Test the data preprocessing functionality."""
    
    def setUp(self):
        self.preprocessor = BookDataPreprocessor()
    
    def test_sample_dataset_creation(self):
        """Test that sample dataset is created correctly."""
        dataset = self.preprocessor._create_sample_dataset()
        
        self.assertIn('train', dataset)
        self.assertIn('validation', dataset)
        self.assertGreater(len(dataset['train']), 0)
        self.assertGreater(len(dataset['validation']), 0)
    
    def test_qa_generation(self):
        """Test Q&A pair generation."""
        text = "The Great Gatsby is a novel by F. Scott Fitzgerald. It was published in 1925."
        title = "The Great Gatsby"
        
        qa_pairs = self.preprocessor.generate_qa_pairs(text, title)
        
        # Should generate at least some QA pairs
        self.assertGreater(len(qa_pairs), 0)
        
        # Check structure of generated pairs
        for qa in qa_pairs:
            self.assertIn('question', qa)
            self.assertIn('answer', qa)
            self.assertIn('context', qa)
            self.assertIn('title', qa)
    
    def test_t5_formatting(self):
        """Test T5 text-to-text formatting."""
        qa_pairs = [
            {
                'question': 'Who wrote this book?',
                'answer': 'F. Scott Fitzgerald',
                'context': 'The Great Gatsby is a novel by F. Scott Fitzgerald.',
                'title': 'The Great Gatsby'
            }
        ]
        
        formatted = self.preprocessor.format_for_t5(qa_pairs)
        
        self.assertEqual(len(formatted), 1)
        self.assertIn('input_text', formatted[0])
        self.assertIn('target_text', formatted[0])
        
        # Check format
        input_text = formatted[0]['input_text']
        self.assertTrue(input_text.startswith('question:'))
        self.assertIn('context:', input_text)


class TestModelConfig(unittest.TestCase):
    """Test model configuration."""
    
    def test_config_creation(self):
        """Test that config is created with correct defaults."""
        config = T5BookQAConfig()
        
        self.assertEqual(config.model_name, "t5-small")
        self.assertGreater(config.max_input_length, 0)
        self.assertGreater(config.max_output_length, 0)
        self.assertGreater(config.num_epochs, 0)
    
    def test_training_args(self):
        """Test training arguments generation."""
        config = T5BookQAConfig()
        args = config.get_training_args("./test_output")
        
        self.assertEqual(args.output_dir, "./test_output")
        self.assertEqual(args.num_train_epochs, config.num_epochs)
        self.assertEqual(args.learning_rate, config.learning_rate)
    
    def test_model_info(self):
        """Test model information function."""
        info = get_model_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("t5-small", info)
        self.assertIn("parameters", info["t5-small"])


class TestEvaluator(unittest.TestCase):
    """Test evaluation functionality."""
    
    def setUp(self):
        self.evaluator = T5BookQAEvaluator()
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        # Test basic normalization
        normalized = self.evaluator.normalize_answer("The Great Gatsby")
        self.assertEqual(normalized, "great gatsby")
        
        # Test punctuation removal
        normalized = self.evaluator.normalize_answer("F. Scott Fitzgerald!")
        self.assertEqual(normalized, "f scott fitzgerald")
    
    def test_exact_match(self):
        """Test exact match scoring."""
        predictions = ["F. Scott Fitzgerald", "1925", "novel"]
        references = ["F. Scott Fitzgerald", "1925", "book"]
        
        score = self.evaluator.exact_match_score(predictions, references)
        
        # Should get 2/3 correct (66.67%)
        self.assertAlmostEqual(score, 2/3, places=2)
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        predictions = ["F. Scott Fitzgerald", "1925", "dystopian novel"]
        references = ["F. Scott Fitzgerald", "1925", "novel"]
        
        f1 = self.evaluator.f1_score(predictions, references)
        
        # Should be > 0 and <= 1
        self.assertGreater(f1, 0)
        self.assertLessEqual(f1, 1)
    
    def test_evaluation_pipeline(self):
        """Test full evaluation pipeline."""
        predictions = ["F. Scott Fitzgerald", "1949", "novel"]
        references = ["F. Scott Fitzgerald", "1949", "dystopian novel"]
        
        results = self.evaluator.evaluate_predictions(predictions, references)
        
        # Check that all expected metrics are present
        expected_metrics = ["exact_match", "f1", "bleu", "rouge1", "rouge2", "rougeL"]
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], (int, float))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_data_processing_pipeline(self):
        """Test the complete data processing pipeline."""
        preprocessor = BookDataPreprocessor()
        
        # Create sample dataset
        dataset = preprocessor._create_sample_dataset()
        
        # Process dataset
        processed = preprocessor.process_dataset(dataset, num_samples=2)
        
        # Verify structure
        self.assertIn('train', processed)
        self.assertIn('validation', processed)
        
        # Check that data was processed correctly
        train_sample = processed['train'][0]
        required_fields = ['input_text', 'target_text', 'question', 'context', 'answer']
        for field in required_fields:
            self.assertIn(field, train_sample)
    
    def test_sample_question_generation(self):
        """Test question generation with known inputs."""
        preprocessor = BookDataPreprocessor()
        
        text = "The Great Gatsby is a novel by F. Scott Fitzgerald. It was published in 1925."
        title = "The Great Gatsby"
        
        qa_pairs = preprocessor.generate_qa_pairs(text, title)
        
        # Should generate author and publication year questions
        questions = [qa['question'] for qa in qa_pairs]
        answers = [qa['answer'] for qa in qa_pairs]
        
        # Check for expected content
        question_text = ' '.join(questions).lower()
        answer_text = ' '.join(answers).lower()
        
        # Should mention the book title or ask about author/year
        self.assertTrue(
            'gatsby' in question_text or 
            'author' in question_text or 
            'published' in question_text
        )


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataPreprocessor,
        TestModelConfig, 
        TestEvaluator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)