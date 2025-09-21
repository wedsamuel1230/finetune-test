# Gemini API Integration for Hybrid Routing

This document explains how to use Google Gemini API in the hybrid routing system.

## Overview

The hybrid routing system now supports Google Gemini API for handling general knowledge questions. When a question is determined to be general knowledge (not book/literature related), it will be routed to the Gemini API for processing.

## Features

- **Automatic Routing**: Questions are automatically classified as domain-specific (books/literature) or general knowledge
- **Gemini Integration**: General knowledge questions are sent to Google Gemini API
- **Fallback Handling**: Graceful fallback when API key is not provided or API calls fail
- **Configurable Models**: Support for different Gemini models (gemini-1.5-flash, gemini-pro, etc.)

## Setup

### 1. Install Dependencies

The Google Generative AI library has been added to requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Set Environment Variable

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or pass it directly in your code:

```python
from examples.hybrid_routing import HybridQARouter

router = HybridQARouter(gemini_api_key="your_api_key_here")
```

## Usage

### Basic Usage

```python
from examples.hybrid_routing import HybridQARouter

# Initialize with Gemini API key
router = HybridQARouter(gemini_api_key="your_api_key")

# Or let it read from environment variable
router = HybridQARouter()  # Reads GEMINI_API_KEY env var

# Ask questions
result = router.route_and_answer("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Model used: {result['model_used']}")
```

### Batch Processing

```python
questions = [
    "Who wrote The Great Gatsby?",      # → SLM (book question)
    "What is the capital of France?",   # → Gemini (general question)
    "What genre is 1984?",             # → SLM (book question)
    "How does photosynthesis work?"     # → Gemini (general question)
]

results = router.batch_answer(questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Model: {result['model_used']}")
    print()
```

### Custom Gemini Model

```python
router = HybridQARouter(
    gemini_api_key="your_api_key",
    gemini_model="gemini-pro"  # Default is "gemini-1.5-flash"
)
```

## Routing Logic

The system automatically routes questions based on content:

### Domain-Specific (→ SLM)
Questions containing book/literature keywords or references:
- Author names (Fitzgerald, Orwell, Austen, etc.)
- Book titles (Gatsby, 1984, Pride and Prejudice, etc.)
- Literary terms (novel, protagonist, genre, etc.)

### General Knowledge (→ Gemini)
All other questions:
- Science questions
- Geography questions  
- Mathematics questions
- Current events
- General factual questions

## Error Handling

The system includes robust error handling:

1. **No API Key**: Falls back to placeholder responses with clear messaging
2. **API Errors**: Catches exceptions and provides fallback responses
3. **Network Issues**: Handles timeouts and connectivity problems gracefully

## Example Output

```
Question: What is the capital of France?
Model: LLM (Google Gemini)
Routing: general_knowledge
Answer: The capital of France is Paris.

Question: Who wrote The Great Gatsby?
Model: SLM (Fine-tuned T5-small)
Routing: domain_specific
Answer: F. Scott Fitzgerald wrote The Great Gatsby.
```

## Testing

Run the test script to verify integration:

```bash
python test_gemini_integration.py
```

Or run the demo:

```bash
python examples/hybrid_routing.py
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gemini_api_key` | `None` | Gemini API key (reads from env if not provided) |
| `gemini_model` | `"gemini-1.5-flash"` | Gemini model to use |
| `slm_model_path` | `None` | Path to fine-tuned SLM model |

## Performance Notes

- Gemini API calls typically take 1-3 seconds
- Domain-specific questions are processed locally (faster)
- Consider implementing caching for repeated questions
- Monitor API usage and costs

## Troubleshooting

### Common Issues

1. **"No Gemini API key provided"**
   - Set the `GEMINI_API_KEY` environment variable
   - Or pass `gemini_api_key` parameter to constructor

2. **API timeout errors**
   - Check internet connectivity
   - Verify API key is valid
   - Try again after a brief wait

3. **Import errors**
   - Ensure `google-generativeai` is installed
   - Run `pip install -r requirements.txt`

### Getting Help

- Check Google AI documentation: https://ai.google.dev/docs
- Verify API key permissions and quotas
- Review logs for detailed error messages