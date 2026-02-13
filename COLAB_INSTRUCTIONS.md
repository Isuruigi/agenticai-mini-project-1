# 🚀 Using Google Colab for Fine-Tuning

## Why Use Colab?

| Feature | Your RTX 3050 | Colab T4 (FREE) |
|---------|---------------|-----------------|
| VRAM | 6GB | 15GB |
| Speed | Good | 30% faster |
| Cost | $0 | $0 |
| Risk | May run out of memory | Plenty of headroom |
| Time Limit | Unlimited | 12 hours (enough!) |

## Quick Start (5 steps)

### 1. Prep Your Files
```bash
# On your PC, zip the datasets
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 1"
Compress-Archive -Path datasets\train.jsonl,datasets\golden_test_set.jsonl -DestinationPath datasets.zip
```

### 2. Open Colab
- Go to: https://colab.research.google.com
- Click: **File → Upload notebook**
- Upload: `notebooks/finetune_colab.ipynb`

### 3. Enable GPU
- Click: **Runtime → Change runtime type**
- Select: **T4 GPU**
- Click: **Save**

### 4. Add Your HF Token
**Option A (Recommended - Secure)**:
- Click the 🔑 icon (left sidebar)
- Click: **Add new secret**
- Name: `HF_TOKEN`
- Value: [paste your Hugging Face token]

**Option B (Quick)**:
- The notebook will prompt you when needed

### 5. Run Everything!
- Click: **Runtime → Run all**
- Upload your `train.jsonl` and `golden_test_set.jsonl` when prompted
- Wait ~1.5-2 hours
- Download `lora_adapters.zip` when done

## After Training

### Download and Install
```bash
# 1. Extract downloaded file on your PC
Expand-Archive -Path Downloads\lora_adapters.zip -DestinationPath "d:\Zuu Crew Agentic AI\Projects\Mini Project 1\models\"

# 2. Run evaluation
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 1"
python utils\evaluate_systems.py

# 3. Launch web UI
python web_ui.py
```

## Advantages

✅ **More VRAM**: 15GB vs 6GB  
✅ **Faster**: T4 optimized for inference  
✅ **No local strain**: Your PC stays cool  
✅ **Free**: No cost  
✅ **Easy**: Just upload & run  

## Disadvantages

⚠️ **Session limit**: 12 hours max (fine-tuning takes ~2 hours, so OK!)  
⚠️ **Upload time**: ~1 minute to upload datasets  
⚠️ **Download time**: ~2 minutes to download model  

## Troubleshooting

### "Runtime disconnected"
- Colab kicked you due to inactivity
- Just reconnect and re-run

### "Out of RAM"
- This shouldn't happen with T4
- If it does, reduce batch_size to 2

### "Can't access Llama-3"
- Make sure you've accepted Meta's terms on Hugging Face
- Check your HF_TOKEN is correct

## Comparison: Local vs Colab

**Use Local (RTX 3050)** if:
- You want full control
- You have time to wait
- You don't mind potential memory issues

**Use Colab (T4)** if:
- You want it to "just work"
- You prefer faster training
- You want guaranteed completion

## What We're Currently Doing

Right now, fine-tuning is running on your **local RTX 3050**. You have two options:

1. **Let it continue** - It should work, just might be slower
2. **Stop it and use Colab** - Guaranteed to work, faster

To switch to Colab:
```bash
# Stop current training (Ctrl+C in terminal)
# Then follow the Quick Start above
```

## Cost Breakdown

| Method | GPU Cost | API Cost | Total |
|--------|----------|----------|-------|
| Local RTX 3050 | $0 | $1.30 (Q&A gen) | **$1.30** |
| Google Colab T4 | $0 | $1.30 (Q&A gen) | **$1.30** |

**Both are equally cheap!** Choose based on convenience.
