"""
Data preprocessor for converting Katharinelw/Book dataset to SQuAD-like format for T5 training.
"""

import pandas as pd
import json
import re
from datasets import load_dataset, Dataset, DatasetDict
from typing import List, Dict, Tuple, Optional
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookDataPreprocessor:
    """Preprocesses book data into SQuAD-like Q&A format for T5 training."""
    
    def __init__(self, max_context_length: int = 512, max_question_length: int = 128):
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        
    def load_book_dataset(self) -> Dataset:
        """Load the Katharinelw/Book dataset."""
        try:
            dataset = load_dataset("Katharinelw/Book")
            logger.info(f"Loaded dataset with {len(dataset['train'])} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to sample data if dataset not available
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> DatasetDict:
        """Create a sample dataset for testing purposes."""
        logger.info("Creating sample dataset for testing")
        sample_data = [
            {
                "text": "The Great Gatsby is a novel by F. Scott Fitzgerald. It was published in 1925 and is set in the summer of 1922. The story follows Nick Carraway, who becomes neighbors with the mysterious Jay Gatsby.",
                "title": "The Great Gatsby"
            },
            {
                "text": "To Kill a Mockingbird is a novel by Harper Lee published in 1960. The story takes place in the fictional town of Maycomb, Alabama, during the 1930s. It follows Scout Finch and her father Atticus.",
                "title": "To Kill a Mockingbird"
            },
            {
                "text": "1984 is a dystopian social science fiction novel by George Orwell. Published in 1949, it presents a totalitarian society ruled by Big Brother. The protagonist Winston Smith works for the Ministry of Truth.",
                "title": "1984"
            }
        ]
        
        return DatasetDict({
            "train": Dataset.from_list(sample_data * 10),  # Multiply for more samples
            "validation": Dataset.from_list(sample_data[:2])
        })
    
    def generate_qa_pairs(self, text: str, title: str) -> List[Dict[str, str]]:
        """Generate Q&A pairs from book text using rule-based approach."""
        qa_pairs = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 20:
                continue
                
            # Generate different types of questions
            qa_pairs.extend(self._generate_questions_for_sentence(sentence, title, sentences))
        
        return qa_pairs
    
    def _generate_questions_for_sentence(self, sentence: str, title: str, context_sentences: List[str]) -> List[Dict[str, str]]:
        """Generate various types of questions for a given sentence."""
        qa_pairs = []
        
        # Question templates based on common patterns
        questions = []
        
        # Who questions
        if any(name in sentence for name in ['is', 'was', 'are', 'were']):
            if 'by' in sentence:
                author_match = re.search(r'by ([A-Z][a-z]+ [A-Z][a-z]+)', sentence)
                if author_match:
                    author = author_match.group(1)
                    questions.append({
                        "question": f"Who wrote {title}?",
                        "answer": author
                    })
        
        # When questions
        year_match = re.search(r'(\d{4})', sentence)
        if year_match:
            year = year_match.group(1)
            questions.append({
                "question": f"When was {title} published?",
                "answer": year
            })
        
        # What questions
        if 'novel' in sentence.lower():
            questions.append({
                "question": f"What type of book is {title}?",
                "answer": "novel"
            })
        
        # Where questions
        location_pattern = r'in ([A-Z][a-z]+(?:, [A-Z][a-z]+)?)'
        location_match = re.search(location_pattern, sentence)
        if location_match:
            location = location_match.group(1)
            questions.append({
                "question": f"Where is {title} set?",
                "answer": location
            })
        
        # Create context from surrounding sentences
        context = sentence
        if len(context_sentences) > 1:
            context = ' '.join(context_sentences[:3])  # Use first 3 sentences as context
        
        # Format for T5 (text-to-text)
        for q in questions:
            qa_pairs.append({
                "context": context[:self.max_context_length],
                "question": q["question"][:self.max_question_length],
                "answer": q["answer"],
                "title": title
            })
        
        return qa_pairs
    
    def format_for_t5(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format Q&A pairs for T5 text-to-text training."""
        formatted_data = []
        
        for qa in qa_pairs:
            # T5 input format: "question: <question> context: <context>"
            input_text = f"question: {qa['question']} context: {qa['context']}"
            target_text = qa['answer']
            
            formatted_data.append({
                "input_text": input_text,
                "target_text": target_text,
                "question": qa['question'],
                "context": qa['context'],
                "answer": qa['answer'],
                "title": qa.get('title', '')
            })
        
        return formatted_data
    
    def process_dataset(self, dataset: DatasetDict, num_samples: Optional[int] = None) -> DatasetDict:
        """Process the full dataset into T5-ready format."""
        processed_data = {"train": [], "validation": []}
        
        for split in ["train", "validation"]:
            if split in dataset:
                split_data = dataset[split]
                if num_samples:
                    split_data = split_data.select(range(min(num_samples, len(split_data))))
                
                for item in split_data:
                    text = item.get('text', '')
                    title = item.get('title', 'Unknown Book')
                    
                    qa_pairs = self.generate_qa_pairs(text, title)
                    formatted_pairs = self.format_for_t5(qa_pairs)
                    processed_data[split].extend(formatted_pairs)
        
        logger.info(f"Generated {len(processed_data['train'])} training samples")
        logger.info(f"Generated {len(processed_data['validation'])} validation samples")
        
        return DatasetDict({
            "train": Dataset.from_list(processed_data["train"]),
            "validation": Dataset.from_list(processed_data["validation"])
        })
    
    def save_dataset(self, dataset: DatasetDict, output_path: str):
        """Save the processed dataset."""
        dataset.save_to_disk(output_path)
        logger.info(f"Dataset saved to {output_path}")
    
    def get_sample_data(self, dataset: DatasetDict, num_samples: int = 5) -> List[Dict]:
        """Get sample data for inspection."""
        return dataset["train"].select(range(min(num_samples, len(dataset["train"]))))


if __name__ == "__main__":
    # Example usage
    preprocessor = BookDataPreprocessor()
    
    # Load and process dataset
    raw_dataset = preprocessor.load_book_dataset()
    processed_dataset = preprocessor.process_dataset(raw_dataset, num_samples=100)
    
    # Show sample
    samples = preprocessor.get_sample_data(processed_dataset)
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Input: {sample['input_text']}")
        print(f"Target: {sample['target_text']}")