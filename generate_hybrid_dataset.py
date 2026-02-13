"""
Hybrid Q&A Generator - Cost Optimized
Uses existing Gemini results + Claude API for remainder
"""

import os
import sys
import json
import glob

sys.path.append('utils')

from qa_generator import create_qa_dataset  # Claude version
from qa_generator_gemini import load_environment


def load_existing_checkpoint():
    """Load the most recent Gemini checkpoint"""
    checkpoint_files = glob.glob('datasets/checkpoint_*.json')
    
    if not checkpoint_files:
        print("No Gemini checkpoints found, starting fresh with Claude")
        return [], 0
    
    # Get latest checkpoint
    latest = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    max_chunk_id = max(item['chunk_id'] for item in data)
    
    print(f"✓ Found Gemini checkpoint: {len(data)} Q&A pairs from {max_chunk_id + 1} chunks")
    
    return data, max_chunk_id + 1


def main():
    print("="*60)
    print("HYBRID Q&A GENERATION (Gemini + Claude)")
    print("="*60)
    
    # Load existing Gemini progress
    existing_data, start_chunk = load_existing_checkpoint()
    
    # Load all chunks
    chunk_files = sorted(glob.glob('data/raw_chunks/chunk_*.txt'))
    target_chunks = 100  # Reduced from 150 to save costs
    
    chunks = []
    for file_path in chunk_files[:target_chunks]:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks.append(f.read())
    
    print(f"\nTarget: {target_chunks} chunks total")
    print(f"Already done (Gemini): {start_chunk} chunks")
    print(f"Remaining (Claude): {target_chunks - start_chunk} chunks")
    
    if start_chunk >= target_chunks:
        print("\n✓ Already have enough data!")
        # Just split and save
        split_idx = int(len(existing_data) * 0.8)
        train_data = existing_data[:split_idx]
        test_data = existing_data[split_idx:]
    else:
        # Generate remaining with Claude
        remaining_chunks = chunks[start_chunk:]
        
        estimated_cost = len(remaining_chunks) * 0.07  # ~$0.07 per chunk
        print(f"\nEstimated Claude cost: ${estimated_cost:.2f}")
        
        confirm = input("\nProceed with Claude API? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Canceled.")
            return
        
        print(f"\nGenerating {len(remaining_chunks)} chunks with Claude...")
        
        # Use original Claude generator
        train_data_new, test_data_new = create_qa_dataset(
            remaining_chunks,
            output_path='datasets/',
            num_chunks=len(remaining_chunks),
            save_interval=10
        )
        
        # Combine with Gemini data
        all_data = existing_data + train_data_new + test_data_new
        
        # Re-split 80/20
        split_idx = int(len(all_data) * 0.8)
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]
    
    # Save final datasets
    train_path = "datasets/train.jsonl"
    test_path = "datasets/golden_test_set.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print("DATASET COMPLETE!")
    print(f"{'='*60}")
    print(f"Total Q&A pairs: {len(train_data) + len(test_data)}")
    print(f"Train: {len(train_data)}")
    print(f"Test: {len(test_data)}")
    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"\n✓ Gemini gave us {len(existing_data)} pairs for FREE")
    print(f"✓ Claude completed the rest")
    print(f"\n🎉 Ready for fine-tuning!")


if __name__ == "__main__":
    main()
