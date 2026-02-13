# API Setup Walkthrough - Operation Ledger-Mind

Follow these steps **in order** to get all your API keys. This should take about 15-20 minutes total.

---

## ✅ Step 1: Hugging Face Account (5 minutes)

### Create Account
1. Go to: https://huggingface.co/join
2. Click **"Sign up"**
3. Enter your email and create a password
4. Verify your email (check inbox/spam)

### Get API Token
1. After logging in, click your **profile picture** (top right)
2. Click **"Settings"**
3. In the left sidebar, click **"Access Tokens"**
4. Click **"New token"**
5. Give it a name: `ledger-mind-project`
6. Select **"Write"** permission (important!)
7. Click **"Generate token"**
8. **COPY THE TOKEN** - it starts with `hf_` (you won't see it again!)
9. Save it somewhere safe (Notepad, password manager, etc.)

### Accept Llama-3 License (CRITICAL!)
1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click **"Agree and access repository"**
3. Fill out Meta's form (name, organization, etc.)
4. Click **"Submit"**
5. Should get instant approval (you'll see a green checkmark)

**✅ You should have:**
- Hugging Face token (starts with `hf_`)
- Access to Llama-3-8B model

---

## ✅ Step 2: Anthropic API (5 minutes)

### Create Account
1. Go to: https://console.anthropic.com/
2. Click **"Sign up"**
3. Enter your email and password
4. Verify your email

### Get API Key
1. After logging in, you'll see the dashboard
2. In the left sidebar, click **"API Keys"**
3. Click **"Create Key"**
4. Give it a name: `ledger-mind`
5. Click **"Create"**
6. **COPY THE API KEY** - it starts with `sk-ant-` (you won't see it again!)
7. Save it somewhere safe

### Check Free Credits
1. In the left sidebar, click **"Billing"**
2. You should see **$5.00 in free credits**
3. This is enough for the entire project (~$3-5 usage)

**✅ You should have:**
- Anthropic API key (starts with `sk-ant-`)
- $5 free credits visible

---

## ✅ Step 3: Weaviate Cloud (5 minutes)

### Create Account
1. Go to: https://console.weaviate.cloud/
2. Click **"Sign up"**
3. **Easiest**: Click "Continue with GitHub" (uses your GitHub account)
   - Or use email/password if you prefer
4. Complete the signup

### Create Free Cluster
1. After logging in, click **"Create Cluster"**
2. Select **"Free Sandbox"** (14-day trial, perfect for this project)
3. Choose a cluster name: `uber-financials`
4. Select region: Choose the one **closest to you** (for faster speed)
   - Example: `US East`, `EU West`, `Asia Pacific`, etc.
5. Click **"Create"**
6. Wait ~2 minutes for cluster to spin up (you'll see a progress bar)

### Get Cluster Details
1. Once ready, click on your cluster name
2. You'll see **Cluster URL** - looks like: `https://uber-financials-xxxxx.weaviate.network`
3. **COPY THIS URL** and save it

### Get API Key
1. In the cluster dashboard, click **"API Keys"** tab
2. Click **"Generate API Key"**
3. **COPY THE KEY** and save it
4. (Optional) Give it a description: `ledger-mind-project`

**✅ You should have:**
- Weaviate cluster URL (starts with `https://`)
- Weaviate API key

---

## ✅ Step 4: Google Colab (2 minutes)

### Access Colab
1. Go to: https://colab.research.google.com/
2. Sign in with your **Google account** (Gmail)
3. That's it! Colab is ready.

### Set Up GPU (You'll do this when running notebooks)
When you open a notebook:
1. Click **"Runtime"** menu → **"Change runtime type"**
2. Under "Hardware accelerator", select **"T4 GPU"**
3. Click **"Save"**

### Optional: Colab Pro
- **NOT required** for this project
- If you want faster training: https://colab.research.google.com/signup
- Cost: $10/month (gives you A100 GPU instead of T4)
- You can decide later after trying free tier

**✅ You should have:**
- Access to Google Colab
- Know how to enable T4 GPU

---

## 📝 Save Your API Keys

Create a **secure text file** with all your keys. You'll need these when running the project.

**Template:**
```
HUGGING FACE
Token: hf_xxxxxxxxxxxxxxxxxxxxx

ANTHROPIC
API Key: sk-ant-xxxxxxxxxxxxxxxxxxxxx

WEAVIATE
Cluster URL: https://uber-financials-xxxxx.weaviate.network
API Key: xxxxxxxxxxxxxxxxxxxxx

GOOGLE COLAB
Email: your-gmail@gmail.com
GPU: T4 (free tier)
```

**⚠️ IMPORTANT:**
- Never share these keys publicly
- Don't commit them to GitHub
- Store them in a password manager or encrypted file

---

## 🧪 Next Steps

Once you have all 4 APIs set up:

1. **Test the environment** - Run verification script to confirm everything works
2. **Create project structure** - Set up folders and files
3. **Begin data factory** - Start processing the PDF

---

## ❓ Troubleshooting

### "Can't find Hugging Face access tokens page"
- Make sure you're logged in
- URL: https://huggingface.co/settings/tokens

### "Llama-3 access denied"
- You must accept the license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- If still denied, wait 5-10 minutes and try again

### "No free credits on Anthropic"
- Check the billing page
- Free credits might be promotional - if not available, you'll need to add $5 credit card

### "Weaviate cluster creation failed"
- Try a different region
- Make sure you selected "Free Sandbox" tier
- Contact Weaviate support if issues persist

### "Colab says no GPU available"
- Free tier has usage limits
- Try again later or upgrade to Colab Pro
- T4 GPU is usually available off-peak hours

---

## ✅ All Done!

Once you have all keys saved, you're ready to proceed with the project setup!

**Total Cost So Far: $0** 🎉

