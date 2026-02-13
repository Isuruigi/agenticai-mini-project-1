# Quick Start Guide - Operation Ledger-Mind

## ✅ Setup Complete!

Your project structure is ready:
```
Mini Project 1/
├── data/                  # PDF and raw chunks
├── datasets/              # train.jsonl, test.jsonl
├── notebooks/             # Jupyter notebooks (4 total)
├── models/
│   └── lora_adapters/     # Fine-tuned model weights
├── utils/                 # Helper Python modules
├── outputs/               # Results and analysis
├── test_environment.py    # Environment verification script
├── requirements.txt       # Python dependencies
├── .env                   # API keys (Linux/Mac)
└── set_env.ps1           # API keys (Windows)
```

## 🚀 Next Steps

### Step 1: Set Environment Variables (Windows)

**Option A: PowerShell (temporary - lasts until you close terminal)**
```powershell
.\set_env.ps1
```

**Option B: Python-dotenv (recommended for notebooks)**
We'll load the `.env` file directly in the notebooks.

### Step 2: Install Python Packages

```powershell
pip install -r requirements.txt
```

This will install:
- Transformers, PEFT, TRL (for fine-tuning)
- Weaviate, Sentence Transformers (for RAG)
- Anthropic, PyPDF (for data generation)
- And more...

**Note:** This may take 5-10 minutes. Some packages are large.

### Step 3: Test Environment

Once packages are installed:
```powershell
# First, set environment variables
.\set_env.ps1

# Then run test
python test_environment.py
```

You should see all tests pass:
```
✓ Imports        : PASS
✓ GPU            : PASS (or WARNING if no GPU)
✓ API Keys       : PASS
✓ Anthropic      : PASS
✓ Hugging Face   : PASS
✓ Weaviate       : PASS
```

## 📝 Important Notes

### GPU Requirement
- This project needs a GPU for fine-tuning
- **Recommended**: Use Google Colab (free T4 GPU)
- **Local**: Only if you have NVIDIA GPU with 12GB+ VRAM

### API Keys Security
- Never commit `.env` or `set_env.ps1` to GitHub
- Already added to `.gitignore`
- Keep your keys private!

### Google Colab Alternative

If you prefer to use Colab instead of local setup:
1. All notebooks will run in Colab
2. Install packages in Colab: `!pip install -r requirements.txt`
3. Set API keys in Colab Secrets (left sidebar 🔑)

## 🎯 What's Next After Setup?

1. ✅ Environment setup (you are here)
2. 📄 Download Uber 2024 Annual Report PDF
3. 🏭 Build Data Factory (generate Q&A pairs)
4. 🎓 Fine-tune The Intern model
5. 📚 Build The Librarian RAG system
6. ⚖️ Evaluate and compare both systems
7. 📊 Write engineering report

## 🆘 Troubleshooting

### "pip is not recognized"
Make sure Python is installed and added to PATH.

### "Package installation failed"
Try: `pip install --upgrade pip` then retry.

### "test_environment.py fails on GPU test"
Expected if no local GPU. You'll use Google Colab for training.

### "Weaviate connection failed"
- Check URL has `https://` prefix
- Verify API key is correct
- Cluster might be sleeping (wait 1-2 minutes)

## 📞 Need Help?

Check the implementation plan for detailed guides on each phase.

---

**Ready to start?** Install the packages and run the environment test!
