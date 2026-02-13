# Operation Ledger-Mind 🤖

> **Comparing Fine-Tuned LLMs vs RAG Systems for Financial Document Analysis**

An end-to-end AI system comparison project analyzing Uber's 2024 Annual Report using:
- **"The Intern"**: Fine-tuned Llama-3 8B with QLoRA
- **"The Librarian"**: Advanced RAG system with hybrid search

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (RTX 3050 or better)
- ~20GB disk space

### Installation

```bash
# 1. Clone/navigate to project
cd "Mini Project 1"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
./set_env.ps1  # Windows
# or
source .env    # Linux/Mac

# 4. Verify setup
python quick_test.py
```

### Usage

**Complete Pipeline** (after Q&A dataset is ready):
```bash
python run_pipeline.py
```

**Individual Components**:
```bash
# Generate Q&A dataset (if not done)
python generate_hybrid_dataset.py

# Fine-tune The Intern
python utils/finetune_intern.py

# Setup The Librarian
python utils/rag_system.py

# Run evaluation
python utils/evaluate_systems.py

# Generate visualizations
python utils/visualizations.py
```

**Web UI** (compare both systems):
```bash
python web_ui.py
# Open http://localhost:7860
```

**CLI Testing**:
```bash
python test_inference.py intern "What was Uber's revenue?"
python test_inference.py librarian "What are the risk factors?"
```

## 📊 Project Structure

```
Mini Project 1/
├── data/
│   ├── Uber_Annual_Report_2024.pdf
│   └── raw_chunks/              # 542 document chunks
│
├── datasets/
│   ├── train.jsonl              # Training data (~800 pairs)
│   └── golden_test_set.jsonl    # Test data (~200 pairs)
│
├── utils/
│   ├── pdf_processor.py         # PDF chunking
│   ├── qa_generator.py          # Claude Q&A generation
│   ├── qa_generator_gemini.py   # Gemini Q&A generation
│   ├── prompt_templates.py      # Generation prompts
│   ├── finetune_intern.py       # Fine-tuning pipeline
│   ├── rag_system.py            # RAG implementation
│   ├── evaluate_systems.py      # Evaluation framework
│   └── visualizations.py        # Chart generation
│
├── models/lora_adapters/        # Fine-tuned weights
│
├── outputs/
│   ├── evaluation_results.csv   # Detailed results
│   ├── evaluation_summary.json  # Summary metrics
│   ├── rouge_comparison.png     # Visualizations
│   ├── latency_comparison.png
│   └── category_breakdown.png
│
├── web_ui.py                    # Gradio interface
├── test_inference.py            # CLI testing
├── run_pipeline.py              # Automation script
├── generate_hybrid_dataset.py   # Hybrid Q&A generation
│
├── .env                         # API keys
├── requirements.txt             # Dependencies
├── SESSION_SUMMARY.md           # Progress tracker
├── QUICK_REFERENCE.md           # Quick guide
└── README.md                    # This file
```

## 🤖 The Systems

### The Intern (Fine-Tuned)
- **Base Model**: Llama-3 8B
- **Method**: QLoRA with 4-bit quantization
- **Training**: ~1,000 Q&A pairs, 3 epochs
- **Strengths**: 
  - Fast inference (local)
  - Learns document style
  - No API calls needed
- **Limitations**:
  - Limited to training data
  - May hallucinate
  - Requires GPU

### The Librarian (RAG)
- **Vector DB**: Weaviate Cloud
- **Embeddings**: Sentence Transformers
- **Search**: Hybrid (vector + BM25)
- **Strengths**:
  - Always grounded in source
  - Covers entire document
  - No hallucinations
  - Easy to update
- **Limitations**:
  - Retrieval quality dependent
  - May miss cross-chunk context

## 📈 Results

Run evaluation to see:
- ROUGE scores (accuracy)
- Latency (speed)
- Cost analysis
- Category breakdown

Results saved to `outputs/` directory.

## 💰 Cost Breakdown

| Component | Cost |
|-----------|------|
| Gemini API (10 chunks) | FREE |
| Claude API (90 chunks) | ~$1.30 |
| Weaviate Cloud | FREE |
| GPU Training | FREE (local) |
| **Total** | **~$1.30** |

## 🛠️ Technical Details

### Hardware Used
- **GPU**: NVIDIA GeForce RTX 3050 (6GB VRAM)
- **RAM**: 16GB+
- **Storage**: ~20GB

### Key Technologies
- PyTorch + Transformers
- PEFT (LoRA/QLoRA)
- Weaviate Vector Database
- Sentence Transformers
- Claude & Gemini APIs
- Gradio for UI

### Training Parameters
- Quantization: 4-bit NF4
- LoRA rank: 16
- Learning rate: 2e-4
- Batch size: 4
- Epochs: 3

## 📚 Documentation

- **[Walkthrough](walkthrough.md)**: Complete technical guide
- **[Quick Reference](QUICK_REFERENCE.md)**: Fast commands
- **[Session Summary](SESSION_SUMMARY.md)**: Progress tracker

## 🎯 Next Steps

1. **Deploy**: Package as Docker container or deploy to Hugging Face Spaces
2. **Optimize**: Tune RAG alpha parameter, try different embedding models
3. **Extend**: Add more documents, create domain-specific fine-tuning
4. **Scale**: Implement batch processing, add caching

## 📝 Example Questions

Try these with either system:

```python
questions = [
    "What was Uber's total revenue in 2024?",
    "What are the main risk factors mentioned?",
    "How does Uber describe its competitive position?",
    "What growth strategies does Uber outline?",
    "What was the CEO's message to shareholders?",
]
```

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different models
- Try other RAG configurations
- Add new evaluation metrics
- Improve the web UI

## 📄 License

Educational use only.

## 🙏 Acknowledgments

- Meta AI for Llama-3
- Anthropic for Claude API
- Google for Gemini API
- Weaviate for vector database
- Hugging Face for transformers

---

**Status**: ✅ 100% Complete

**Last Updated**: 2026-02-13


