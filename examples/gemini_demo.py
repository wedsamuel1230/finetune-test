#!/usr/bin/env python3
"""
Simple demo of Gemini API integration in hybrid routing system.
This shows how the system works with and without API keys.
"""

from hybrid_routing import HybridQARouter

def demo_without_api_key():
    """Demonstrate system behavior without API key."""
    print("🔄 Demo: Hybrid Routing WITHOUT Gemini API Key")
    print("=" * 55)
    
    router = HybridQARouter()
    
    questions = [
        "Who wrote The Great Gatsby?",
        "What is the capital of France?",
        "What type of novel is 1984?",
        "How does photosynthesis work?"
    ]
    
    for question in questions:
        result = router.route_and_answer(question)
        print(f"\nQ: {question}")
        print(f"Routed to: {result['model_used']}")
        print(f"Decision: {result['routing_decision']}")
        if len(result['answer']) > 80:
            print(f"Answer: {result['answer'][:80]}...")
        else:
            print(f"Answer: {result['answer']}")

def demo_with_mock_key():
    """Demonstrate system behavior with mock API key."""
    print("\n🔧 Demo: Hybrid Routing WITH Mock Gemini API Key")
    print("=" * 55)
    
    # This will attempt to initialize Gemini but fail gracefully
    router = HybridQARouter(gemini_api_key="mock_key_for_demo")
    
    questions = [
        "What is machine learning?",
        "Who is the protagonist in To Kill a Mockingbird?",
        "What is quantum computing?",
        "What genre is Pride and Prejudice?"
    ]
    
    for question in questions:
        result = router.route_and_answer(question)
        print(f"\nQ: {question}")
        print(f"Routed to: {result['model_used']}")
        print(f"Decision: {result['routing_decision']}")
        if len(result['answer']) > 80:
            print(f"Answer: {result['answer'][:80]}...")
        else:
            print(f"Answer: {result['answer']}")

def show_routing_stats():
    """Show routing statistics."""
    print("\n📊 Routing Statistics Demo")
    print("=" * 30)
    
    router = HybridQARouter()
    
    all_questions = [
        "Who wrote The Great Gatsby?",
        "What is the capital of France?", 
        "What type of novel is 1984?",
        "How does photosynthesis work?",
        "Who is the protagonist in To Kill a Mockingbird?",
        "What is machine learning?",
        "What genre is Pride and Prejudice?",
        "How do you solve a quadratic equation?"
    ]
    
    stats = router.get_routing_stats(all_questions)
    
    print(f"Total questions: {stats['total_questions']}")
    print(f"📚 Book/Literature questions: {stats['domain_questions']} ({stats['domain_percentage']:.1f}%)")
    print(f"🌍 General knowledge questions: {stats['general_questions']}")
    
    print(f"\nBook/Literature questions will be routed to SLM:")
    for q in all_questions:
        if router.is_domain_question(q):
            print(f"  • {q}")
    
    print(f"\nGeneral knowledge questions will be routed to Gemini:")
    for q in all_questions:
        if not router.is_domain_question(q):
            print(f"  • {q}")

def main():
    """Main demo function."""
    print("🎯 Gemini API Integration Demo")
    print("=" * 40)
    print("This demo shows how the hybrid routing system")
    print("integrates with Google Gemini API for general questions.")
    print()
    
    # Demo without API key
    demo_without_api_key()
    
    # Demo with mock API key
    demo_with_mock_key()
    
    # Show routing statistics
    show_routing_stats()
    
    print("\n" + "=" * 60)
    print("🚀 To use with real Gemini API:")
    print("1. Get API key: https://makersuite.google.com/app/apikey")
    print("2. export GEMINI_API_KEY='your_api_key_here'")
    print("3. python examples/hybrid_routing.py")
    print()
    print("📖 See GEMINI_INTEGRATION.md for full documentation")

if __name__ == "__main__":
    main()