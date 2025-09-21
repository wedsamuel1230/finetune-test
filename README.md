# T5 Fine-tuning for Book Question Answering

This repository contains a complete pipeline for fine-tuning T5-small model on the Katharinelw/Book dataset for domain-specific question answering, formatted according to The Stanford Question Answering Dataset (SQuAD) standard.

## Features

- **Generative Q&A**: Fine-tune T5-small for generating answers to book-related questions
- **SQuAD-compatible format**: Data preprocessing to match SQuAD standards
- **Google Colab ready**: Optimized for training in Google Colab environment
- **Hybrid system integration**: Route domain questions to SLM, general to LLM
- **Comprehensive evaluation**: BLEU, ROUGE metrics for generative QA

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Quick Demo (No GPU Required)

```bash
python demo.py                    # Run all component demos
python demo.py --component data   # Test data processing only
```

### 3. Google Colab Training (Recommended) 🚀

**Enhanced for Google Colab with full automation:**

1. **Open the notebook**: [`notebooks/t5_book_qa_training.ipynb`](notebooks/t5_book_qa_training.ipynb) in Google Colab
2. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU (T4)
3. **Run all cells**: The notebook now automatically:
   - Detects Colab environment
   - Installs dependencies
   - Clones repository
   - Mounts Google Drive for model persistence
   - Optimizes settings for T4 GPU
   - Saves trained model to Google Drive

**New Colab Features:**
- ✅ **Auto-detection**: Automatically configures for Colab vs local environment
- ✅ **Google Drive integration**: Models persist across sessions
- ✅ **Memory optimization**: T4 GPU-optimized batch sizes and memory management
- ✅ **File upload support**: Upload custom datasets directly in the notebook
- ✅ **Model download**: Easy download of trained models
- ✅ **Error handling**: Graceful fallbacks for dataset loading issues
- ✅ **Runtime restart handling**: Robust session management

### 4. Local Training

```bash
# Quick setup validation
python setup_and_test.py

# Train with subset first (recommended)
python train.py --subset_first

# Full training with custom parameters
python train.py --num_epochs 5 --batch_size 4 --learning_rate 1e-4
```

### 5. Quick API Usage

```python
from src.trainer import T5BookQATrainer

trainer = T5BookQATrainer()
trainer.prepare_data()
trainer.train()
```

## Project Structure

```
├── src/
│   ├── data_preprocessor.py    # Data loading and SQuAD formatting
│   ├── model_config.py         # T5 model configuration
│   ├── trainer.py              # Training pipeline
│   └── evaluator.py            # Evaluation metrics
├── notebooks/
│   └── t5_book_qa_training.ipynb  # Google Colab notebook
├── examples/
│   ├── hybrid_routing.py       # SLM/LLM routing example
│   └── inference_demo.py       # Model inference demo
└── tests/
    └── test_pipeline.py        # Basic testing
```

## Training Process

1. **Task Definition**: Generative Q&A for book domain using T5-small
2. **Data Preparation**: Convert book dataset to SQuAD-like Q&A pairs
3. **Model Setup**: Configure T5-small with text-to-text format
4. **Training**: Start with subset, monitor loss/overfitting
5. **Evaluation**: BLEU/ROUGE metrics and human spot-checks
6. **Integration**: Hybrid routing for domain vs. general questions

## License

MIT License - see LICENSE file for details.