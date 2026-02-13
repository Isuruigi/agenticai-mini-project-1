"""
Q&A Generation Module for Operation Ledger-Mind
Generates question-answer pairs from document chunks using Claude API
"""

import os
import re
import json
import time
from typing import List, Dict, Tuple
from anthropic import Anthropic
from tqdm import tqdm

from prompt_templates import (
    QUESTION_GENERATION_PROMPT,
    ANSWER_GENERATION_PROMPT,
    categorize_question
)


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


def generate_questions(chunk: str, client: Anthropic, max_retries: int = 3) -> List[str]:
    """
    Generate 10 questions from a chunk using Claude
    
    Args:
        chunk: Text chunk to generate questions from
        client: Anthropic client
        max_retries: Number of retry attempts on failure
        
    Returns:
        List of 10 questions
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.8,  # Creative questions
                messages=[{
                    "role": "user",
                    "content": QUESTION_GENERATION_PROMPT.format(chunk=chunk)
                }]
            )
            
            # Parse numbered list
            text = response.content[0].text
            questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', text, re.DOTALL)
            questions = [q.strip() for q in questions if q.strip()]
            
            # Ensure we have exactly 10 questions
            if len(questions) < 10:
                print(f"⚠ Only got {len(questions)} questions, retrying...")
                continue
            
            return questions[:10]  # Take first 10
            
        except Exception as e:
            print(f"✗ Error generating questions (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    return []


def generate_answer(chunk: str, question: str, client: Anthropic, max_retries: int = 3) -> str:
    """
    Generate answer to a question based on chunk context
    
    Args:
        chunk: Context chunk
        question: Question to answer
        client: Anthropic client
        max_retries: Number of retry attempts
        
    Returns:
        Answer text
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.0,  # Deterministic answers
                messages=[{
                    "role": "user",
                    "content": ANSWER_GENERATION_PROMPT.format(
                        chunk=chunk,
                        question=question
                    )
                }]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"✗ Error generating answer (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    return ""


def create_qa_dataset(
    chunks: List[str],
    output_path: str = 'datasets/',
    num_chunks: int = None,
    save_interval: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create Q&A dataset from chunks
    
    Args:
        chunks: List of text chunks
        output_path: Directory to save datasets
        num_chunks: Number of chunks to process (None = all)
        save_interval: Save checkpoint every N chunks
        
    Returns:
        Tuple of (train_data, test_data)
    """
    # Load API key
    load_environment()
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    # Prepare output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Select chunks to process
    if num_chunks:
        chunks = chunks[:num_chunks]
    
    print(f"\n{'='*60}")
    print(f"Q&A GENERATION")
    print(f"{'='*60}")
    print(f"Processing {len(chunks)} chunks")
    print(f"Target: {len(chunks) * 10} Q&A pairs\n")
    
    dataset = []
    failed_chunks = []
    
    # Process each chunk
    for idx, chunk in enumerate(tqdm(chunks, desc="Generating Q&A pairs")):
        try:
            # Generate questions
            questions = generate_questions(chunk, client)
            
            if not questions:
                print(f"\n⚠ Chunk {idx}: No questions generated, skipping")
                failed_chunks.append(idx)
                continue
            
            # Generate answers for each question
            for question in questions:
                try:
                    answer = generate_answer(chunk, question, client)
                    
                    dataset.append({
                        "instruction": question,
                        "input": "",  # Empty for fine-tuning format
                        "output": answer,
                        "chunk_id": idx,
                        "category": categorize_question(question)
                    })
                    
                except Exception as e:
                    print(f"\n✗ Error with question '{question[:50]}...': {e}")
            
            # Save checkpoint
            if (idx + 1) % save_interval == 0:
                checkpoint_path = f"{output_path}checkpoint_{idx+1}.json"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Checkpoint saved: {len(dataset)} Q&A pairs")
            
            # Rate limiting
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"\n✗ Error processing chunk {idx}: {e}")
            failed_chunks.append(idx)
            continue
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Q&A pairs: {len(dataset)}")
    if failed_chunks:
        print(f"Failed chunks: {failed_chunks}")
    
    # Calculate category distribution
    category_counts = {}
    for item in dataset:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nCategory Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Split into train/test (80/20)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_data)} pairs")
    print(f"  Test: {len(test_data)} pairs")
    
    # Save datasets
    train_path = f"{output_path}train.jsonl"
    test_path = f"{output_path}golden_test_set.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    
    return train_data, test_data


if __name__ == "__main__":
    # Test the Q&A generation pipeline
    print("=" * 60)
    print("Q&A GENERATION TEST")
    print("=" * 60)
    
    # Load chunks
    import glob
    chunk_files = sorted(glob.glob('data/raw_chunks/chunk_*.txt'))[:10]  # First 10
    
    chunks = []
    for file_path in chunk_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks.append(f.read())
    
    print(f"Loaded {len(chunks)} chunks for testing\n")
    
    # Generate Q&A pairs
    train_data, test_data = create_qa_dataset(
        chunks,
        output_path='datasets/',
        num_chunks=10,  # Test with 10 chunks
        save_interval=5
    )
    
    print("\n" + "=" * 60)
    print("SAMPLE Q&A PAIRS")
    print("=" * 60)
    
    # Show samples from each category
    for category in ['Hard Fact', 'Strategic', 'Stylistic']:
        samples = [item for item in train_data if item['category'] == category][:2]
        print(f"\n{category} Examples:")
        for i, sample in enumerate(samples, 1):
            print(f"\n  Q{i}: {sample['instruction']}")
            print(f"  A{i}: {sample['output'][:100]}...")
    
    print("\n✓ Q&A generation test complete!")
