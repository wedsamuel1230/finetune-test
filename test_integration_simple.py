#!/usr/bin/env python3
"""
Simple test to verify Gemini integration is working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples'))

from hybrid_routing import HybridQARouter

def test_integration():
    """Test that the integration is properly set up."""
    print("🧪 Testing Gemini API Integration Structure")
    print("=" * 50)
    
    # Test 1: Without API key
    print("\n1. Testing without API key (should use fallback):")
    router1 = HybridQARouter()
    result1 = router1.route_and_answer("What is 2+2?")
    print(f"   Model: {result1['model_used']}")
    print(f"   Result: {'✅ PASS' if 'Google Gemini' in result1['model_used'] else '❌ FAIL'}")
    
    # Test 2: With mock API key (should handle gracefully)
    print("\n2. Testing with mock API key (should handle gracefully):")
    router2 = HybridQARouter(gemini_api_key="mock_key_for_demo")
    result2 = router2.route_and_answer("What is machine learning?")
    print(f"   Model: {result2['model_used']}")
    print(f"   Result: {'✅ PASS' if 'Google Gemini' in result2['model_used'] else '❌ FAIL'}")
    
    # Test 3: Routing logic
    print("\n3. Testing routing logic:")
    test_cases = [
        ("Who wrote The Great Gatsby?", True, "Book question"),
        ("What is the capital of France?", False, "General question"),
        ("What genre is 1984?", True, "Book question with title"),
        ("How does photosynthesis work?", False, "Science question")
    ]
    
    routing_passed = 0
    for question, expected_domain, description in test_cases:
        is_domain = router1.is_domain_question(question)
        status = "✅ PASS" if is_domain == expected_domain else "❌ FAIL"
        print(f"   {description}: {status}")
        if is_domain == expected_domain:
            routing_passed += 1
    
    # Test 4: API structure
    print("\n4. Testing API integration structure:")
    has_gemini_import = True
    try:
        import google.generativeai as genai
        print("   Google Generative AI import: ✅ PASS")
    except ImportError:
        print("   Google Generative AI import: ❌ FAIL")
        has_gemini_import = False
    
    # Test 5: Method availability
    print("\n5. Testing method availability:")
    methods_to_check = ['_init_gemini', 'answer_with_llm', 'route_and_answer']
    method_tests = 0
    for method in methods_to_check:
        has_method = hasattr(router1, method)
        status = "✅ PASS" if has_method else "❌ FAIL"
        print(f"   {method}: {status}")
        if has_method:
            method_tests += 1
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Routing logic: {routing_passed}/{len(test_cases)} tests passed")
    print(f"   Method tests: {method_tests}/{len(methods_to_check)} tests passed")
    print(f"   API integration: {'✅ Ready' if has_gemini_import else '❌ Missing dependencies'}")
    
    total_score = routing_passed + method_tests + (1 if has_gemini_import else 0)
    total_possible = len(test_cases) + len(methods_to_check) + 1
    print(f"   Overall: {total_score}/{total_possible} ({total_score/total_possible*100:.1f}%)")
    
    if total_score == total_possible:
        print("\n🎉 All tests passed! Gemini integration is ready.")
        print("   To use with real API key:")
        print("   export GEMINI_API_KEY='your_key_here'")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    test_integration()