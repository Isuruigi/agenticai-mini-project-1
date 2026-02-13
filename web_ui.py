"""
Web UI for Operation Ledger-Mind
Compare The Intern vs The Librarian side-by-side
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
import os


# Global variables for loaded models
intern_model = None
intern_tokenizer = None
librarian_client = None
librarian_embedder = None


def load_environment():
    """Load API keys from .env file"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'export' in line:
                    parts = line.replace('export ', '').split('=', 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip().strip('"')
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")


def load_intern():
    """Load The Intern (fine-tuned model)"""
    global intern_model, intern_tokenizer
    
    if intern_model is not None:
        return "The Intern already loaded!"
    
    try:
        load_environment()
        
        # Load base model with 4-bit quantization
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
        intern_model = PeftModel.from_pretrained(base_model, "models/lora_adapters")
        intern_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            token=os.getenv('HF_TOKEN')
        )
        
        return "✓ The Intern loaded successfully!"
    except Exception as e:
        return f"✗ Error loading The Intern: {e}\nMake sure fine-tuning is complete."


def load_librarian():
    """Load The Librarian (RAG system)"""
    global librarian_client, librarian_embedder
    
    if librarian_client is not None:
        return "The Librarian already loaded!"
    
    try:
        load_environment()
        
        # Connect to Weaviate
        weaviate_url = os.getenv('WEAVIATE_URL')
        weaviate_key = os.getenv('WEAVIATE_API_KEY')
        
        if not weaviate_url.startswith('http'):
            weaviate_url = f'https://{weaviate_url}'
        
        librarian_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key),
            skip_init_checks=True
        )
        
        # Load embedder
        librarian_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        return "✓ The Librarian loaded successfully!"
    except Exception as e:
        return f"✗ Error loading The Librarian: {e}\nMake sure RAG system is set up."


def query_intern(question):
    """Query The Intern"""
    if intern_model is None or intern_tokenizer is None:
        return "⚠️ Please load The Intern first!"
    
    prompt = f"""### Instruction:
{question}

### Response:
"""
    
    inputs = intern_tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = intern_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = intern_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response


def query_librarian(question):
    """Query The Librarian"""
    if librarian_client is None or librarian_embedder is None:
        return "⚠️ Please load The Librarian first!"
    
    try:
        # Generate query embedding
        query_vector = librarian_embedder.encode(question).tolist()
        
        # Get collection
        collection = librarian_client.collections.get("UberFinancialDocs")
        
        # Hybrid search
        response = collection.query.hybrid(
            query=question,
            vector=query_vector,
            alpha=0.7,  # 70% vector, 30% keyword
            limit=3,
            return_metadata=['score']
        )
        
        # Format results
        context_parts = []
        for obj in response.objects:
            context_parts.append(obj.properties["content"][:300])
        
        context = "\n\n".join(context_parts)
        
        # For now, return context (can add LLM generation later)
        return f"**Retrieved Context:**\n\n{context}\n\n*(Note: Add LLM for natural language answer)*"
        
    except Exception as e:
        return f"✗ Error: {e}"


def compare_both(question):
    """Query both systems and return side-by-side results"""
    intern_answer = query_intern(question)
    librarian_answer = query_librarian(question)
    
    return intern_answer, librarian_answer


# Create Gradio interface
with gr.Blocks(title="Operation Ledger-Mind", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Operation Ledger-Mind
    ## Compare "The Intern" (Fine-tuned) vs "The Librarian" (RAG)
    
    Ask questions about Uber's 2024 Annual Report and see how both AI systems respond!
    """)
    
    with gr.Row():
        with gr.Column():
            load_intern_btn = gr.Button("🔄 Load The Intern", variant="primary")
            intern_status = gr.Textbox(label="The Intern Status", interactive=False)
        
        with gr.Column():
            load_librarian_btn = gr.Button("🔄 Load The Librarian", variant="primary")
            librarian_status = gr.Textbox(label="The Librarian Status", interactive=False)
    
    gr.Markdown("---")
    
    question_input = gr.Textbox(
        label="Your Question",
        placeholder="e.g., What was Uber's revenue in 2024?",
        lines=2
    )
    
    with gr.Row():
        intern_btn = gr.Button("Ask The Intern", variant="secondary")
        librarian_btn = gr.Button("Ask The Librarian", variant="secondary")
        compare_btn = gr.Button("Compare Both", variant="primary")
    
    with gr.Row():
        intern_output = gr.Textbox(label="The Intern's Answer", lines=10)
        librarian_output = gr.Textbox(label="The Librarian's Answer", lines=10)
    
    gr.Markdown("""
    ### Example Questions:
    - What was Uber's total revenue in 2024?
    - What are the main risk factors mentioned?
    - How does Uber describe its competitive position?
    - What growth strategies does Uber outline?
    """)
    
    # Button actions
    load_intern_btn.click(load_intern, outputs=intern_status)
    load_librarian_btn.click(load_librarian, outputs=librarian_status)
    
    intern_btn.click(query_intern, inputs=question_input, outputs=intern_output)
    librarian_btn.click(query_librarian, inputs=question_input, outputs=librarian_output)
    compare_btn.click(compare_both, inputs=question_input, outputs=[intern_output, librarian_output])


if __name__ == "__main__":
    print("="*60)
    print("OPERATION LEDGER-MIND - WEB UI")
    print("="*60)
    print("\nStarting Gradio interface...")
    print("Access at: http://localhost:7860")
    print("\nNote: Load models first before querying!")
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
