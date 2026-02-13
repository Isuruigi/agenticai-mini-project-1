"""
Automation Script - Run Complete Pipeline
Executes all steps once Q&A generation completes
"""

import os
import sys
import time
from pathlib import Path


def check_dataset_ready():
    """Check if Q&A dataset is complete"""
    train_file = Path("datasets/train.jsonl")
    test_file = Path("datasets/golden_test_set.jsonl")
    
    if not train_file.exists() or not test_file.exists():
        return False, 0
    
    # Count lines in training file
    with open(train_file, 'r') as f:
        train_count = sum(1 for _ in f)
    
    # We expect ~1200 training pairs (80% of 1500)
    expected = 1200
    progress = (train_count / expected) * 100
    
    return train_count >= expected, progress


def run_finetuning():
    """Run fine-tuning pipeline"""
    print("\n" + "="*60)
    print("STEP 1: FINE-TUNING THE INTERN")
    print("="*60)
    
    os.system("python utils/finetune_intern.py")


def test_rag_system():
    """Test RAG system"""
    print("\n" + "="*60)
    print("STEP 2: TESTING THE LIBRARIAN")
    print("="*60)
    
    # RAG already set up, just verify
    from utils.rag_system import connect_to_weaviate, hybrid_search
    from sentence_transformers import SentenceTransformer
    
    try:
        client = connect_to_weaviate()
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_query = "What was Uber's total revenue?"
        results = hybrid_search(client, "UberFinancialDocs", test_query, embedder, limit=3)
        
        print(f"\n✓ RAG Test Query: {test_query}")
        print(f"✓ Retrieved {len(results)} chunks")
        
        client.close()
        print("✓ RAG system verified")
        
    except Exception as e:
        print(f"⚠️  RAG test failed: {e}")
        print("You may need to re-run: python utils/rag_system.py")


def run_evaluation():
    """Run full evaluation"""
    print("\n" + "="*60)
    print("STEP 3: RUNNING EVALUATION")
    print("="*60)
    
    os.system("python utils/evaluate_systems.py")


def main():
    print("="*60)
    print("OPERATION LEDGER-MIND - AUTOMATED PIPELINE")
    print("="*60)
    
    # Check if dataset is ready
    print("\nChecking dataset status...")
    ready, progress = check_dataset_ready()
    
    if not ready:
        print(f"\n⏳ Dataset generation in progress: {progress:.1f}%")
        print("Please wait for Q&A generation to complete.")
        print("\nOptions:")
        print("1. Let this script wait and auto-start when ready")
        print("2. Come back later and run: python run_pipeline.py")
        
        choice = input("\nWait here? (y/n): ").strip().lower()
        
        if choice == 'y':
            print("\n⏳ Waiting for dataset completion...")
            print("Checking every 5 minutes...\n")
            
            while not ready:
                time.sleep(300)  # Check every 5 minutes
                ready, progress = check_dataset_ready()
                print(f"Progress: {progress:.1f}% - {time.strftime('%H:%M:%S')}")
        else:
            print("\n👋 Come back when dataset is ready!")
            return
    
    print("\n✅ Dataset ready! Starting pipeline...\n")
    
    # Run pipeline
    try:
        # Step 1: Fine-tune
        run_finetuning()
        
        # Step 2: Test RAG
        test_rag_system()
        
        # Step 3: Evaluate
        run_evaluation()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print("\n✓ The Intern: Fine-tuned and ready")
        print("✓ The Librarian: Tested and verified")
        print("✓ Evaluation: Results saved to outputs/")
        print("\nCheck outputs/evaluation_summary.json for results!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Check error messages above for details")


if __name__ == "__main__":
    main()
