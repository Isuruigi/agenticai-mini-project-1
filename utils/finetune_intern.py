"""
Fine-Tuning Pipeline for "The Intern" - Llama-3 8B Model
Uses QLoRA (4-bit quantization + LoRA) for efficient training on consumer GPU
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import json


def load_environment():
    """Load API keys from .env file"""
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'export' in line:
                    parts = line.replace('export ', '').split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip('"')
                        os.environ[key] = value
                        env_vars[key] = value
        return env_vars
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
        return {}


def format_instruction(sample):
    """Format data into instruction format for fine-tuning"""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""


def setup_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B"):
    """
    Load model with 4-bit quantization for efficient training
    """
    print(f"Loading model: {model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.getenv('HF_TOKEN')
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv('HF_TOKEN')
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print("✓ Model loaded with 4-bit quantization")
    return model, tokenizer


def setup_lora_config():
    """
    Configure LoRA for efficient fine-tuning
    """
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("✓ LoRA configuration set")
    return lora_config


def prepare_dataset(train_file="datasets/train.jsonl", test_file="datasets/golden_test_set.jsonl"):
    """
    Load and prepare dataset for training
    """
    print(f"Loading dataset from {train_file}")
    
    # Load datasets
    dataset = load_dataset('json', data_files={
        'train': train_file,
        'test': test_file
    })
    
    print(f"✓ Loaded {len(dataset['train'])} training samples")
    print(f"✓ Loaded {len(dataset['test'])} test samples")
    
    return dataset


def train_model(
    model,
    tokenizer,
    dataset,
    lora_config,
    output_dir="models/lora_adapters",
    num_epochs=3,
    batch_size=4
):
    """
    Fine-tune the model using QLoRA
    """
    print("\nStarting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=lora_config,
        dataset_text_field="text",  # Will be created by formatting function
        formatting_func=lambda x: format_instruction(x),
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )
    
    # Train
    print("\n" + "="*60)
    print("TRAINING IN PROGRESS")
    print("="*60)
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"\n✓ Model saved to {output_dir}")
    
    return trainer


def test_model(model, tokenizer, test_prompt="What was Uber's revenue in 2024?"):
    """
    Test the fine-tuned model with a sample question
    """
    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)
    
    formatted_prompt = f"""### Instruction:
{test_prompt}

### Response:
"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    print(f"\nQuestion: {test_prompt}")
    print(f"\nAnswer: {response}")
    
    return response


if __name__ == "__main__":
    print("="*60)
    print("THE INTERN - FINE-TUNING PIPELINE")
    print("="*60)
    
    # Load environment
    load_environment()
    
    # Check if dataset exists
    if not os.path.exists("datasets/train.jsonl"):
        print("\n⚠️  Training dataset not found!")
        print("Q&A generation is still running. This script will be ready once it completes.")
        print("\nTo run fine-tuning later:")
        print("  python utils/finetune_intern.py")
        exit(0)
    
    # Setup
    print("\n1. Loading Model...")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("\n2. Configuring LoRA...")
    lora_config = setup_lora_config()
    
    print("\n3. Loading Dataset...")
    dataset = prepare_dataset()
    
    print("\n4. Starting Training...")
    trainer = train_model(model, tokenizer, dataset, lora_config)
    
    print("\n5. Testing Model...")
    test_model(model, tokenizer)
    
    print("\n" + "="*60)
    print("✓ FINE-TUNING COMPLETE!")
    print("="*60)
    print("\nModel saved to: models/lora_adapters/")
    print("Ready for evaluation!")
