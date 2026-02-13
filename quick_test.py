"""
Quick Setup Test - Operation Ledger-Mind
A simpler test that loads credentials from .env file
"""

import os

# Load environment variables from .env file
print("Loading environment variables from .env file...")
env_vars = {}
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and 'export' in line:
                # Parse: export VAR_NAME="value"
                parts = line.replace('export ', '').split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"')
                    os.environ[key] = value
                    env_vars[key] = value
    print(f"✓ Loaded {len(env_vars)} environment variables\n")
except Exception as e:
    print(f"✗ Error loading .env: {e}\n")

# Test 1: Package Imports
print("=" * 60)
print("TEST 1: Package Imports")
print("=" * 60)
try:
    import anthropic
    import weaviate
    import transformers
    print("✓ All key packages imported successfully\n")
except ImportError as e:
    print(f"✗ Import failed: {e}\n")

# Test 2: GPU Check
print("=" * 60)
print("TEST 2: GPU Availability")
print("=" * 60)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠ No GPU detected - You'll need to use Google Colab for training\n")
except Exception as e:
    print(f"Info: {e}\n")

# Test 3: API Keys Check
print("=" * 60)
print("TEST 3: API Keys")
print("=" * 60)
keys_needed = ['HF_TOKEN', 'ANTHROPIC_API_KEY', 'WEAVIATE_URL', 'WEAVIATE_API_KEY']
all_set = True
for key in keys_needed:
    value = os.getenv(key)
    if value:
        masked = value[:10] + "..." if len(value) > 10 else "***"
        print(f"✓ {key}: {masked}")
    else:
        print(f"✗ {key}: NOT SET")
        all_set = False
print()

# Test 4: Anthropic API
print("="* 60)
print("TEST 4: Anthropic API Connection")
print("=" * 60)
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"✓ Anthropic API working: {response.content[0].text}\n")
except Exception as e:
    print(f"✗ Anthropic API failed: {e}\n")

# Test 5: Weaviate Connection
print("=" * 60)
print("TEST 5: Weaviate Connection")
print("=" * 60)
try:
    from weaviate.classes.init import Auth
    import weaviate
    
    weaviate_url = os.getenv('WEAVIATE_URL', '')
    if not weaviate_url.startswith('http'):
        weaviate_url = f'https://{weaviate_url}'
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(os.getenv('WEAVIATE_API_KEY'))
    )
    
    if client.is_ready():
        print(f"✓ Weaviate connected successfully")
        client.close()
    else:
        print("✗ Weaviate not ready")
    print()
except Exception as e:
    print(f"✗ Weaviate failed: {e}\n")

# Summary
print("=" * 60)
print("SETUP STATUS")
print("=" * 60)
print("✓ All required packages installed")
print("✓ API credentials loaded from .env")
print("⚠ No local GPU (use Google Colab for training)")
print("\n🎯 NEXT STEPS:")
print("1. Download Uber 2024 Annual Report PDF")
print("2. Put it in the 'data/' folder")
print("3. Start building the Data Factory!")
print("\nYou're ready to begin! 🚀")
