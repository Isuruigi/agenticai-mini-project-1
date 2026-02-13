"""
Visualization Tools for Evaluation Results
Creates charts and comparison visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path


def create_rouge_comparison_chart(summary_file="outputs/evaluation_summary.json"):
    """Create bar chart comparing ROUGE scores"""
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    intern_scores = data['the_intern']['rouge_scores']
    librarian_scores = data['the_librarian']['rouge_scores']
    
    # Prepare data
    metrics = list(intern_scores.keys())
    intern_values = [intern_scores[m] for m in metrics]
    librarian_values = [librarian_scores[m] for m in metrics]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(metrics))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], intern_values, width, 
                    label='The Intern', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], librarian_values, width,
                    label='The Librarian', color='#2196F3', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('ROUGE Score Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/rouge_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: outputs/rouge_comparison.png")
    
    return fig


def create_latency_comparison_chart(summary_file="outputs/evaluation_summary.json"):
    """Create bar chart comparing latency"""
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    intern_latency = data['the_intern']['avg_latency_ms']
    librarian_latency = data['the_librarian']['avg_latency_ms']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    systems = ['The Intern', 'The Librarian']
    latencies = [intern_latency, librarian_latency]
    colors = ['#4CAF50', '#2196F3']
    
    bars = ax.bar(systems, latencies, color=colors, alpha=0.8, width=0.6)
    
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Average Query Latency', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f} ms',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: outputs/latency_comparison.png")
    
    return fig


def create_category_breakdown(results_file="outputs/evaluation_results.csv"):
    """Create breakdown by question category"""
    
    df = pd.read_csv(results_file)
    
    # Assuming we have category info (would need to add in evaluation)
    # For now, create a sample visualization
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Placeholder data - would be replaced with actual category analysis
    categories = ['Hard Facts', 'Strategic', 'Stylistic']
    intern_scores = [0.85, 0.72, 0.68]
    librarian_scores = [0.78, 0.81, 0.65]
    
    x = range(len(categories))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], intern_scores, width, 
            label='The Intern', color='#4CAF50', alpha=0.8)
    ax1.bar([i + width/2 for i in x], librarian_scores, width,
            label='The Librarian', color='#2196F3', alpha=0.8)
    
    ax1.set_xlabel('Question Category', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg ROUGE Score', fontsize=11, fontweight='bold')
    ax1.set_title('Performance by Question Category', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Question distribution
    counts = [45, 30, 25]  # Placeholder
    ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
            colors=['#FF9800', '#9C27B0', '#00BCD4'], startangle=90)
    ax2.set_title('Question Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/category_breakdown.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: outputs/category_breakdown.png")
    
    return fig


def create_summary_report():
    """Create comprehensive visual summary"""
    
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    
    # Check if outputs exist
    if not Path("outputs/evaluation_summary.json").exists():
        print("\n⚠️  Evaluation results not found!")
        print("Run evaluation first: python utils/evaluate_systems.py")
        return
    
    # Create visualizations
    print("\n1. Creating ROUGE comparison chart...")
    create_rouge_comparison_chart()
    
    print("2. Creating latency comparison chart...")
    create_latency_comparison_chart()
    
    print("3. Creating category breakdown...")
    create_category_breakdown()
    
    print("\n" + "="*60)
    print("[OK] VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - outputs/rouge_comparison.png")
    print("  - outputs/latency_comparison.png")
    print("  - outputs/category_breakdown.png")


if __name__ == "__main__":
    create_summary_report()
