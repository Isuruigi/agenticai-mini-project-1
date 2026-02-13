"""
Test Gemini API and list available models
"""
import os
import google.generativeai as genai

# Load environment
env_vars = {}
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and 'export' in line:
            parts = line.replace('export ', '').split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"')
                os.environ[key] = value

gemini_key = os.getenv('GEMINI_API_KEY')
print(f"API Key: {gemini_key[:20]}...")

# Configure and list models
genai.configure(api_key=gemini_key)

print("\nAvailable Models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")

# Test with gemini-pro
print("\nTesting with gemini-pro...")
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Say hello in 3 words")
print(f"Response: {response.text}")
print("\n✓ Gemini API working!")
