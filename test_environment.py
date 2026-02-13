#!/usr/bin/env python3
"""
Environment Setup Verification Script for Operation Ledger-Mind
Run this BEFORE starting the project to ensure everything works.
"""

import sys
import os

def test_imports():
    """Test all required packages can be imported"""
    print("Testing package imports...")
    try:
        import transformers
        import datasets
        import peft
        import trl
        import bitsandbytes
        import accelerate
        import weaviate
        import sentence_transformers
        import anthropic
        import pypdf
        import pandas
        import rouge_score
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nTo fix this, run:")
        print("pip install transformers datasets peft trl bitsandbytes accelerate")
        print("pip install weaviate-client sentence-transformers rank-bm25")
        print("pip install anthropic pypdf pandas rouge-score")
        return False

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ No GPU available - fine-tuning will be slow or fail!")
            print("  Note: If using Colab, make sure to enable T4 GPU in runtime settings")
            return False
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False

def test_api_keys():
    """Test API keys are configured"""
    print("\nTesting API keys...")
    
    keys = {
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'WEAVIATE_URL': os.getenv('WEAVIATE_URL'),
        'WEAVIATE_API_KEY': os.getenv('WEAVIATE_API_KEY')
    }
    
    all_set = True
    for key, value in keys.items():
        if value:
            masked_value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"✓ {key} is set ({masked_value})")
        else:
            print(f"✗ {key} is NOT set")
            all_set = False
    
    return all_set

def test_anthropic_api():
    """Test Anthropic API connection"""
    print("\nTesting Anthropic API...")
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"✓ Anthropic API working: {response.content[0].text[:20]}...")
        return True
    except Exception as e:
        print(f"✗ Anthropic API failed: {e}")
        return False

def test_hf_auth():
    """Test Hugging Face authentication"""
    print("\nTesting Hugging Face authentication...")
    try:
        from huggingface_hub import login, whoami
        login(token=os.getenv('HF_TOKEN'))
        user = whoami()
        print(f"✓ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"✗ HF authentication failed: {e}")
        return False

def test_weaviate():
    """Test Weaviate connection"""
    print("\nTesting Weaviate connection...")
    try:
        import weaviate
        from weaviate.classes.init import Auth
        
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_key = os.getenv('WEAVIATE_API_KEY')
        
        # Ensure URL has https://
        if not weaviate_url.startswith('http'):
            weaviate_url = f'https://{weaviate_url}'
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key)
        )
        
        if client.is_ready():
            print(f"✓ Connected to Weaviate successfully")
            client.close()
            return True
        else:
            print("✗ Weaviate connection failed")
            return False
    except Exception as e:
        print(f"✗ Weaviate connection failed: {e}")
        return False

def main():
    print("="*60)
    print("OPERATION LEDGER-MIND: Environment Test")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'GPU': test_gpu(),
        'API Keys': test_api_keys(),
        'Anthropic': test_anthropic_api(),
        'Hugging Face': test_hf_auth(),
        'Weaviate': test_weaviate()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test:15} : {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems go! You're ready to start the project.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
