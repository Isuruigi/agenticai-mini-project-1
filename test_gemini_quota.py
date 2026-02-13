"""
Quick test with gemini-pro to check quota
"""
import os
import google.generativeai as genai

# Load environment
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and 'GEMINI' in line:
            parts = line.replace('export ', '').split('=', 1)
            if len(parts) == 2:
                os.environ[parts[0].strip()] = parts[1].strip().strip('"')

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Try gemini-pro (older model with potentially different quota)
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello")
    print(f"✅ gemini-pro WORKS!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ gemini-pro failed: {e}")
    
# Try gemini-2.5-flash (newest)
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say hello")
    print(f"\n✅ gemini-2.5-flash WORKS!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"\n❌ gemini-2.5-flash failed: {e}")
