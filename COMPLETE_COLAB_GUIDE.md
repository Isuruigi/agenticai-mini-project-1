# 🚀 COMPLETE GOOGLE COLAB FINE-TUNING GUIDE

## ✅ Everything You Need - One Place

---

## 📋 PART 1: What You Need (Checklist)

- ✅ Google account (for Colab)
- ✅ Hugging Face account (already have it)
- ✅ Dataset files (already generated)
- ✅ Internet connection
- ✅ 2 hours of time

---

## 🔑 PART 2: Your API Keys & Credentials

### Hugging Face Token
```
your_hf_token_here
```

**Where it's stored**: `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\.env`

**Used for**: Downloading Llama-3 8B model from Hugging Face

---

## 📁 PART 3: Files You Need to Upload

**Location on your PC**: `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\datasets\`

**Files to upload** (when Colab asks):
1. `train.jsonl` (~125 KB - 792 training pairs)
2. `golden_test_set.jsonl` (~32 KB - 198 test pairs)

**How to find them**:
```powershell
# Open File Explorer
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 1\datasets"
# You'll see these 2 files
```

---

## 🎯 PART 4: Complete Step-by-Step Instructions

### Step 1: Open Google Colab
1. Open browser
2. Go to: **https://colab.research.google.com**
3. Sign in with your Google account

### Step 2: Upload the Notebook
1. In Colab, click: **File → Upload notebook**
2. Click: **Choose file**
3. Navigate to: `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\notebooks\`
4. Select: `finetune_colab.ipynb`
5. Click: **Open**

**Expected**: Notebook opens in Colab with title "Operation Ledger-Mind: Fine-Tuning on Google Colab"

### Step 3: Enable T4 GPU (CRITICAL!)
1. Click: **Runtime** (top menu)
2. Click: **Change runtime type**
3. Under "Hardware accelerator", select: **T4 GPU**
4. Click: **Save**

**Expected**: You'll see "T4 GPU" in the top-right corner

### Step 4: Add Hugging Face Token (Secure Method)
1. Click the **🔑 (key) icon** on the LEFT sidebar
2. Click: **+ Add new secret**
3. In **Name** field, type: `HF_TOKEN`
4. In **Value** field, paste: `your_hf_token_here`
5. Toggle **"Notebook access"** to ON (blue)
6. Click: **OK** or **Save**

**Expected**: You'll see "HF_TOKEN" listed under "Secrets"

### Step 5: Run All Cells
1. Click: **Runtime** (top menu)
2. Click: **Run all**
3. **IMPORTANT**: When the upload dialog appears (cell #4), click **Choose Files**
4. Navigate to: `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\datasets\`
5. Select BOTH files:
   - `train.jsonl`
   - `golden_test_set.jsonl`
6. Click: **Open**

**Expected**: Files upload in ~30 seconds

### Step 6: Wait for Training to Complete
**Timeline**:
- Install packages: 2-3 min ✓
- Download Llama-3: 5-10 min ✓
- **Training Epoch 1/3**: 30-40 min ⏳
- **Training Epoch 2/3**: 30-40 min ⏳
- **Training Epoch 3/3**: 30-40 min ⏳
- Save model: 1-2 min ✓

**Total**: ~2 hours

**What you'll see**:
```
Epoch 1/3: [████████████░░░░] 65% | Loss: 1.234
Epoch 2/3: [██████░░░░░░░░░] 40% | Loss: 0.987
Epoch 3/3: [████████████████] 100% | Loss: 0.845
✓ Training Complete!
```

### Step 7: Download the Trained Model
1. When training completes, the last cell will create `lora_adapters.zip`
2. A download will **start automatically**
3. Save to: `Downloads\lora_adapters.zip`

**Expected**: ZIP file ~50-100 MB

---

## 💻 PART 5: After Training - Use the Model Locally

### Step 1: Extract the Model
```powershell
# Open PowerShell
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 1"

# Extract the downloaded model
Expand-Archive -Path "$env:USERPROFILE\Downloads\lora_adapters.zip" -DestinationPath "models\" -Force
```

**Expected**: Folder `models\lora_adapters\` with files inside

### Step 2: Run Evaluation
```powershell
cd "d:\Zuu Crew Agentic AI\Projects\Mini Project 1"
python utils\evaluate_systems.py
```

**Expected**: 
- Evaluation runs for ~30 minutes
- Creates `outputs\evaluation_results.csv`
- Creates `outputs\evaluation_summary.json`

### Step 3: Generate Visualizations
```powershell
python utils\visualizations.py
```

**Expected**:
- Creates `outputs\rouge_comparison.png`
- Creates `outputs\latency_comparison.png`
- Creates `outputs\category_breakdown.png`

### Step 4: Launch Web UI (Test Both Systems)
```powershell
python web_ui.py
```

**Expected**:
- Opens browser at: http://localhost:7860
- You can chat with "The Intern" and "The Librarian" side-by-side!

---

## 🔍 PART 6: What If Something Goes Wrong?

### Issue 1: "Cannot access Llama-3"
**Solution**: Make sure you accepted Meta's license
1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B
2. Click: **Agree and access repository**
3. Re-run the notebook

### Issue 2: "GPU not available"
**Solution**: Colab is busy
1. Wait 5-10 minutes
2. Click: **Runtime → Restart runtime**
3. Click: **Runtime → Run all** again

### Issue 3: "Session crashed"
**Solution**: Rare but possible
1. **Don't worry** - Colab auto-saves
2. Click: **Runtime → Restart runtime**
3. Click: **Runtime → Run all**
4. It will resume from where it left off

### Issue 4: Upload widget doesn't work
**Solution**: Re-run the cell
1. Click on the upload cell (#4)
2. Press: **Ctrl+Enter**
3. The "Choose Files" button will activate

### Issue 5: "Out of memory"
**Solution**: Shouldn't happen with T4, but if it does:
1. In the notebook, find this line in cell #10:
   ```python
   per_device_train_batch_size=4,
   ```
2. Change it to:
   ```python
   per_device_train_batch_size=2,
   ```
3. Re-run from that cell

---

## 📊 PART 7: What Happens During Training

### You'll See These Messages (Normal):

```
✓ Libraries imported
✓ Model loaded with 4-bit quantization
✓ LoRA configured
✓ Loaded 792 training samples
✓ Loaded 198 test samples
✓ Trainer created
```

### Then Training Starts:

```
============================================================
TRAINING STARTED
============================================================
This will take ~1.5-2 hours on T4 GPU
You can close this tab - training will continue!
============================================================

Epoch 1/3:
Step 50/200 | Loss: 1.456 | LR: 0.0002
Step 100/200 | Loss: 1.234 | LR: 0.00018
Step 150/200 | Loss: 1.123 | LR: 0.00016
Step 200/200 | Loss: 1.089 | LR: 0.00014

Epoch 2/3:
Step 50/200 | Loss: 0.987 | LR: 0.00012
...

Epoch 3/3:
Step 200/200 | Loss: 0.845 | LR: 0.00002

✓ Training Complete!
✓ Model saved to lora_adapters/
```

### Loss Values (What's Good):
- **Epoch 1**: Loss ~1.5-1.0 (learning basics)
- **Epoch 2**: Loss ~1.0-0.8 (improving)
- **Epoch 3**: Loss ~0.8-0.6 (final polish)

**Lower is better!** If you see loss going down, training is working correctly.

---

## ⏱️ PART 8: Complete Timeline

| Step | What | Time | Can You Leave? |
|------|------|------|----------------|
| Setup GPU & upload notebook | Manual work | 3 min | No - you need to click |
| Add HF_TOKEN secret | Manual work | 1 min | No - you need to type |
| Run all cells | Click button | 5 sec | No - not yet |
| Upload dataset files | Select files | 30 sec | No - almost done! |
| Install packages | Automated | 2-3 min | Yes! ✓ |
| Download model | Automated | 5-10 min | Yes! ✓ |
| **Training** | **Automated** | **1.5-2 hrs** | **Yes! ✓** |
| Save & download | Automated | 2 min | Yes! ✓ |
| **TOTAL** | | **~2 hours** | |

**You only need to be present for the first ~5 minutes!**

---

## 💰 PART 9: Cost Breakdown

| Resource | Cost |
|----------|------|
| Google Colab T4 GPU | **FREE** |
| Hugging Face (model download) | **FREE** |
| Dataset (already generated) | $1.30 (already paid) |
| **TOTAL** | **$1.30** (already spent) |

**This fine-tuning costs you $0!** 🎉

---

## 🎓 PART 10: After Everything is Done

### You'll Have:
1. ✅ Fine-tuned "Intern" model (in `models/lora_adapters/`)
2. ✅ RAG "Librarian" system (already built)
3. ✅ Evaluation results (ROUGE scores, latency)
4. ✅ Visualization charts (PNG files)
5. ✅ Web UI to test both systems

### What's Left:
1. Create 4 Jupyter notebooks (~1.5 hours)
2. Write engineering report (~2-3 hours)
3. Package & submit (~1 hour)

**You're 75% done!** Just final documentation remains.

---

## 📞 Quick Reference

**Colab Website**: https://colab.research.google.com

**HF Token**: `your_hf_token_here`

**Files to Upload**:
- `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\datasets\train.jsonl`
- `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\datasets\golden_test_set.jsonl`

**Download Goes To**: `%USERPROFILE%\Downloads\lora_adapters.zip`

**Extract To**: `d:\Zuu Crew Agentic AI\Projects\Mini Project 1\models\`

---

## ✅ FINAL CHECKLIST

Before you start:
- [ ] Browser open to colab.research.google.com
- [ ] Notebook file ready: `notebooks\finetune_colab.ipynb`
- [ ] Dataset files ready: `datasets\train.jsonl` and `datasets\golden_test_set.jsonl`
- [ ] HF Token ready: `your_hf_token_here`
- [ ] 2 hours of available time (but you only need to be present for 5 min!)

After training:
- [ ] Model downloaded: `lora_adapters.zip`
- [ ] Model extracted to: `models\lora_adapters\`
- [ ] Evaluation run: `python utils\evaluate_systems.py`
- [ ] Visualizations created: `python utils\visualizations.py`
- [ ] Web UI tested: `python web_ui.py`

---

**NOW YOU'RE READY! START HERE**: https://colab.research.google.com 🚀

Good luck! The hard part is done - this is just clicking through the steps! 💪
