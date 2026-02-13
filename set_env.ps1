# Python Environment Variables for Windows
# Run this in PowerShell to set environment variables

$env:HF_TOKEN = "your_hf_token_here"
$env:ANTHROPIC_API_KEY = "your_anthropic_key_here"
$env:GEMINI_API_KEY = "your_gemini_key_here"
$env:WEAVIATE_URL = "your_weaviate_url_here"
$env:WEAVIATE_API_KEY = "your_weaviate_key_here"

Write-Host "✓ Environment variables set successfully!" -ForegroundColor Green
Write-Host "These are temporary and will reset when you close PowerShell" -ForegroundColor Yellow
