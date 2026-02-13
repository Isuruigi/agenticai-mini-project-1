"""
Full Dataset Generation Script
Generates Q&A pairs from 150 chunks for the complete training dataset
"""

import sys
sys.path.append('utils')

from qa_generator import create_qa_dataset
import glob

# Load chunks
chunk_files = sorted(glob.glob('data/raw_chunks/chunk_*.txt'))[:150]

chunks = []
for file_path in chunk_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks.append(f.read())

print(f"Loaded {len(chunks)} chunks")
print(f"Target: ~{len(chunks) * 10} Q&A pairs")
print(f"\nThis will take approximately {len(chunks) * 50 / 60:.0f} minutes")
print("API cost estimate: $5-7\n")

input("Press Enter to start generation...")

# Generate full dataset
train_data, test_data = create_qa_dataset(
    chunks,
    output_path='datasets/',
    num_chunks=150,
    save_interval=25  # Save checkpoint every 25 chunks
)

print("\n🎉 Full dataset generation complete!")
print(f"Total Q&A pairs: {len(train_data) + len(test_data)}")
print(f"Ready for fine-tuning!")
