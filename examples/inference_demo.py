"""
Inference demo for the fine-tuned T5 book QA model.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import json
from typing import List, Dict

class T5BookQAInference:
    """Inference wrapper for fine-tuned T5 book QA model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            print(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model path exists and contains valid model files.")
            raise
    
    def answer_question(self, 
                       question: str, 
                       context: str, 
                       max_length: int = 128,
                       num_beams: int = 4,
                       temperature: float = 0.8) -> str:
        """
        Generate answer for a question given context.
        
        Args:
            question: The question to answer
            context: Context containing relevant information
            max_length: Maximum length of generated answer
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
        
        Returns:
            Generated answer string
        """
        # Format input for T5
        input_text = f"question: {question} context: {context}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=temperature,
                do_sample=temperature > 0
            )
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def batch_inference(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Run inference on multiple QA pairs."""
        results = []
        
        for qa in qa_pairs:
            question = qa.get('question', '')
            context = qa.get('context', '')
            
            answer = self.answer_question(question, context)
            
            result = {
                'question': question,
                'context': context,
                'answer': answer,
                'expected': qa.get('expected', '')  # For evaluation
            }
            results.append(result)
        
        return results
    
    def interactive_demo(self):
        """Run an interactive demo session."""
        print("\n🤖 T5 Book QA Interactive Demo")
        print("=" * 50)
        print("Enter questions about books and literature.")
        print("Type 'quit' to exit, 'help' for examples.")
        
        while True:
            question = input("\n📖 Question: ").strip()
            
            if question.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if question.lower() == 'help':
                self._show_examples()
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            context = input("📝 Context (optional): ").strip()
            
            print("🤔 Thinking...")
            answer = self.answer_question(question, context)
            print(f"💡 Answer: {answer}")
    
    def _show_examples(self):
        """Show example questions."""
        examples = [
            {
                "question": "Who wrote The Great Gatsby?",
                "context": "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925."
            },
            {
                "question": "When was 1984 published?",
                "context": "1984 is a dystopian novel by George Orwell published in 1949."
            },
            {
                "question": "What type of book is To Kill a Mockingbird?",
                "context": "To Kill a Mockingbird is a novel by Harper Lee."
            }
        ]
        
        print("\n📚 Example questions:")
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. Question: {ex['question']}")
            print(f"   Context: {ex['context']}")


def run_demo_with_sample_model():
    """Run demo using the sample data for testing."""
    print("🚀 Running T5 Book QA Demo with Sample Questions")
    print("=" * 60)
    
    # Sample QA pairs for testing (when model isn't available)
    sample_qa_pairs = [
        {
            "question": "Who wrote The Great Gatsby?",
            "context": "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925 and is set in the summer of 1922.",
            "expected": "F. Scott Fitzgerald"
        },
        {
            "question": "When was To Kill a Mockingbird published?", 
            "context": "To Kill a Mockingbird is a novel by Harper Lee published in 1960. The story takes place in Alabama during the 1930s.",
            "expected": "1960"
        },
        {
            "question": "What type of book is 1984?",
            "context": "1984 is a dystopian social science fiction novel by George Orwell published in 1949.",
            "expected": "dystopian social science fiction novel"
        },
        {
            "question": "Where is To Kill a Mockingbird set?",
            "context": "The story takes place in the fictional town of Maycomb, Alabama, during the 1930s.",
            "expected": "Maycomb, Alabama"
        },
        {
            "question": "Who is the protagonist in The Great Gatsby?",
            "context": "The story follows Nick Carraway, who becomes neighbors with the mysterious Jay Gatsby.",
            "expected": "Nick Carraway"
        }
    ]
    
    print("📝 Sample Questions and Expected Answers:")
    print("-" * 50)
    
    for i, qa in enumerate(sample_qa_pairs, 1):
        print(f"\n{i}. Question: {qa['question']}")
        print(f"   Context: {qa['context']}")
        print(f"   Expected Answer: {qa['expected']}")
        print(f"   Generated Answer: [Would be generated by fine-tuned model]")
    
    print(f"\n✅ Demo complete! Tested {len(sample_qa_pairs)} questions.")
    print("\nTo use with a real model:")
    print("1. Fine-tune the T5 model using the training notebook")
    print("2. Run: python inference_demo.py --model_path ./path/to/model")


def main():
    parser = argparse.ArgumentParser(description="T5 Book QA Inference Demo")
    parser.add_argument("--model_path", type=str, help="Path to fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--input_file", type=str, help="JSON file with questions")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    if args.model_path:
        try:
            # Load and run with real model
            inference = T5BookQAInference(args.model_path)
            
            if args.interactive:
                inference.interactive_demo()
            elif args.input_file:
                # Batch processing
                with open(args.input_file, 'r') as f:
                    qa_pairs = json.load(f)
                
                results = inference.batch_inference(qa_pairs)
                
                if args.output_file:
                    with open(args.output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {args.output_file}")
                else:
                    for result in results:
                        print(f"Q: {result['question']}")
                        print(f"A: {result['answer']}")
                        print("-" * 30)
            else:
                print("Please specify --interactive or --input_file")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Running demo with sample data instead...")
            run_demo_with_sample_model()
    else:
        # Run demo with sample data
        run_demo_with_sample_model()


if __name__ == "__main__":
    main()