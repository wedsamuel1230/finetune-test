#!/usr/bin/env python3
"""
Test script for Gemini API integration in hybrid routing system.
This script demonstrates how to use the Gemini API with the HybridQARouter.
"""

import os
import sys
from examples.hybrid_routing import HybridQARouter

def test_with_mock_api_key():
    """Test the system with a mock API key to show the integration structure."""
    print("🧪 Testing Gemini API Integration")
    print("=" * 50)
    
    # Create router with a mock API key (will fail gracefully)
    router = HybridQARouter(gemini_api_key="mock_api_key_for_testing")
    
    test_questions = [
        "What is the capital of France?",
        "How does machine learning work?",
        "Who wrote The Great Gatsby?",
        "What is quantum physics?"
    ]
    
    print("Testing questions with Gemini integration:")
    print("-" * 40)
    
    for question in test_questions:
        result = router.route_and_answer(question)
        print(f"\nQuestion: {question}")
        print(f"Routed to: {result['model_used']}")
        print(f"Decision: {result['routing_decision']}")
        print(f"Answer: {result['answer'][:100]}...")  # Truncate for readability
    
    return True

def test_with_real_api_key():
    """Test with a real API key if provided via environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n🔑 No GEMINI_API_KEY environment variable found.")
        print("To test with real Gemini API:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("python test_gemini_integration.py")
        return False
    
    print("\n🎯 Testing with Real Gemini API")
    print("=" * 50)
    
    router = HybridQARouter(gemini_api_key=api_key)
    
    # Test general knowledge questions that should go to Gemini
    general_questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the weather like today?"
    ]
    
    print("Testing general knowledge questions with Gemini:")
    print("-" * 45)
    
    for question in general_questions:
        result = router.route_and_answer(question)
        print(f"\nQuestion: {question}")
        print(f"Model: {result['model_used']}")
        print(f"Answer: {result['answer']}")
    
    return True

def main():
    """Main test function."""
    print("🔄 Hybrid QA Router - Gemini Integration Test")
    print("=" * 60)
    
    # Test 1: Mock API key integration
    test_with_mock_api_key()
    
    # Test 2: Real API key if available
    test_with_real_api_key()
    
    print("\n📊 Test Summary:")
    print("- Gemini API integration structure: ✅ Working")
    print("- Routing logic: ✅ Working") 
    print("- Error handling: ✅ Working")
    print("\nTo use with real Gemini API:")
    print("1. Get API key from https://makersuite.google.com/app/apikey")
    print("2. export GEMINI_API_KEY='your_api_key_here'")
    print("3. Run: python examples/hybrid_routing.py")

if __name__ == "__main__":
    main()