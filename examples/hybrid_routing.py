"""
Hybrid routing system example - routes domain questions to SLM, general to LLM.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class HybridQARouter:
    """
    Hybrid system that routes questions to appropriate models:
    - Book/literature questions -> Fine-tuned T5 SLM
    - General questions -> Large Language Model (API)
    """
    
    def __init__(self, slm_model_path: str = None):
        self.slm_model = None
        self.slm_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Domain keywords for routing
        self.book_keywords = [
            'book', 'novel', 'author', 'wrote', 'published', 'character', 
            'story', 'plot', 'chapter', 'setting', 'protagonist', 'literature',
            'fiction', 'read', 'reader', 'narrative', 'literary', 'writing',
            'publish', 'publication', 'genre', 'classic', 'bestseller'
        ]
        
        if slm_model_path:
            self.load_slm(slm_model_path)
    
    def load_slm(self, model_path: str):
        """Load the fine-tuned SLM model."""
        logger.info(f"Loading SLM from {model_path}")
        try:
            self.slm_tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.slm_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.slm_model.to(self.device)
            logger.info("SLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SLM: {e}")
            raise
    
    def is_domain_question(self, question: str, threshold: float = 0.3) -> bool:
        """
        Determine if question is book/literature related.
        
        Args:
            question: The input question
            threshold: Minimum keyword ratio to classify as domain question
        
        Returns:
            True if question is likely book-related
        """
        question_words = question.lower().split()
        keyword_matches = sum(1 for word in question_words if word in self.book_keywords)
        
        if len(question_words) == 0:
            return False
        
        keyword_ratio = keyword_matches / len(question_words)
        return keyword_ratio >= threshold
    
    def answer_with_slm(self, question: str, context: str = "") -> str:
        """Answer using the fine-tuned Small Language Model."""
        if self.slm_model is None or self.slm_tokenizer is None:
            return "SLM not loaded. Please load the model first."
        
        # Format input for T5
        input_text = f"question: {question} context: {context}"
        
        # Tokenize
        inputs = self.slm_tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.slm_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.8
            )
        
        answer = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer_with_llm(self, question: str, context: str = "") -> str:
        """
        Answer using Large Language Model (placeholder for API integration).
        
        In a real implementation, this would call APIs like:
        - OpenAI GPT-4
        - Anthropic Claude
        - Google Gemini
        - etc.
        """
        # Placeholder response
        return f"[LLM Response] I would need to integrate with a large language model API to answer: '{question}'"
    
    def route_and_answer(self, question: str, context: str = "") -> Dict[str, str]:
        """
        Route question to appropriate model and return answer with metadata.
        
        Args:
            question: The question to answer
            context: Optional context for the question
        
        Returns:
            Dictionary with answer, model used, confidence, and reasoning
        """
        is_domain = self.is_domain_question(question)
        
        if is_domain:
            answer = self.answer_with_slm(question, context)
            return {
                "answer": answer,
                "model_used": "SLM (Fine-tuned T5-small)",
                "routing_decision": "domain_specific",
                "confidence": "high" if context else "medium",
                "reasoning": "Question identified as book/literature related"
            }
        else:
            answer = self.answer_with_llm(question, context)
            return {
                "answer": answer,
                "model_used": "LLM (General purpose)",
                "routing_decision": "general_knowledge", 
                "confidence": "medium",
                "reasoning": "Question identified as general knowledge"
            }
    
    def batch_answer(self, questions: List[str], contexts: List[str] = None) -> List[Dict[str, str]]:
        """Answer multiple questions in batch."""
        if contexts is None:
            contexts = [""] * len(questions)
        
        results = []
        for question, context in zip(questions, contexts):
            result = self.route_and_answer(question, context)
            results.append(result)
        
        return results
    
    def get_routing_stats(self, questions: List[str]) -> Dict[str, int]:
        """Get statistics on how questions would be routed."""
        domain_count = sum(1 for q in questions if self.is_domain_question(q))
        general_count = len(questions) - domain_count
        
        return {
            "total_questions": len(questions),
            "domain_questions": domain_count,
            "general_questions": general_count,
            "domain_percentage": (domain_count / len(questions)) * 100 if questions else 0
        }


def demo_hybrid_system():
    """Demonstrate the hybrid routing system."""
    print("🔄 Hybrid QA System Demo")
    print("=" * 50)
    
    # Initialize router (without loading SLM for demo)
    router = HybridQARouter()
    
    # Test questions
    test_questions = [
        "Who wrote The Great Gatsby?",
        "What is the capital of France?", 
        "What type of novel is 1984?",
        "How does photosynthesis work?",
        "Who is the protagonist in To Kill a Mockingbird?",
        "What is the weather like today?",
        "What genre is Pride and Prejudice?",
        "How do you solve a quadratic equation?"
    ]
    
    # Test contexts (for book questions)
    test_contexts = [
        "The Great Gatsby is a novel by F. Scott Fitzgerald published in 1925.",
        "",
        "1984 is a dystopian social science fiction novel by George Orwell.",
        "",
        "To Kill a Mockingbird follows Scout Finch and her father Atticus.",
        "",
        "Pride and Prejudice is a romantic novel by Jane Austen.",
        ""
    ]
    
    # Route questions
    print("Routing Decisions:")
    print("-" * 30)
    
    for question, context in zip(test_questions, test_contexts):
        result = router.route_and_answer(question, context)
        print(f"\nQuestion: {question}")
        print(f"Model: {result['model_used']}")
        print(f"Routing: {result['routing_decision']}")
        print(f"Answer: {result['answer']}")
    
    # Show routing statistics
    stats = router.get_routing_stats(test_questions)
    print(f"\n📊 Routing Statistics:")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Domain questions: {stats['domain_questions']} ({stats['domain_percentage']:.1f}%)")
    print(f"General questions: {stats['general_questions']}")


if __name__ == "__main__":
    demo_hybrid_system()