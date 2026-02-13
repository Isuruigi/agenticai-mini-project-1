"""
Simple Inference Script
Test The Intern and The Librarian with quick queries
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os


def load_environment():
    """Load API keys"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'export' in line:
                    parts = line.replace('export ', '').split('=', 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip().strip('"')
    except:
        pass


def test_intern(question: str):
    """Test The Intern with a question"""
    print("\n" + "="*60)
    print("THE INTERN (Fine-Tuned Model)")
    print("="*60)
    
    load_environment()
    
    print("Loading model...")
    
    # Load with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv('HF_TOKEN')
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, "models/lora_adapters")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=os.getenv('HF_TOKEN')
    )
    
    print("✓ Model loaded")
    
    # Generate answer
    prompt = f"""### Instruction:
{question}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"\nQuestion: {question}")
    print("\nGenerating answer...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    print(f"\nAnswer:\n{response}")
    
    return response


def test_librarian(question: str):
    """Test The Librarian with a question"""
    print("\n" + "="*60)
    print("THE LIBRARIAN (RAG System)")
    print("="*60)
    
    from utils.rag_system import connect_to_weaviate, hybrid_search
    from sentence_transformers import SentenceTransformer
    
    print("Connecting to Weaviate...")
    client = connect_to_weaviate()
    
    print("Loading embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"\nQuestion: {question}")
    print("\nSearching...")
    
    results = hybrid_search(client, "UberFinancialDocs", question, embedder, limit=3, alpha=0.7)
    
    print(f"\n✓ Found {len(results)} relevant chunks:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Chunk {i} (score: {result['score']:.3f}) ---")
        print(result['content'][:300] + "...")
    
    client.close()
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_inference.py [intern|librarian] 'your question'")
        print("\nExample:")
        print("  python test_inference.py intern 'What was Uber\\'s revenue?'")
        print("  python test_inference.py librarian 'What are the risk factors?'")
        exit(1)
    
    system = sys.argv[1].lower()
    question = sys.argv[2]
    
    if system == "intern":
        test_intern(question)
    elif system == "librarian":
        test_librarian(question)
    else:
        print("Error: System must be 'intern' or 'librarian'")
        exit(1)
