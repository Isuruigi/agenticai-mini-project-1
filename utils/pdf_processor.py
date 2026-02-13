"""
PDF Processor Module for Operation Ledger-Mind
Handles PDF loading, cleaning, and chunking
"""

from pypdf import PdfReader
import re
from typing import List


def load_and_clean_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF and remove headers/footers
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Cleaned text content
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        print(f"Loading PDF: {pdf_path}")
        print(f"Total pages: {len(reader.pages)}")
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            
            # Remove common headers/footers patterns
            # Customize these patterns based on the actual PDF format
            page_text = re.sub(r'Uber Technologies.*?Page \d+', '', page_text, flags=re.IGNORECASE)
            page_text = re.sub(r'Page \d+ of \d+', '', page_text)
            page_text = re.sub(r'^\d+$', '', page_text, flags=re.MULTILINE)  # Remove page numbers
            
            text += page_text
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1} pages...")
        
        print(f"✓ PDF loaded successfully")
        print(f"Total characters: {len(text):,}")
        
        return text
        
    except FileNotFoundError:
        print(f"✗ Error: PDF file not found at {pdf_path}")
        raise
    except Exception as e:
        print(f"✗ Error processing PDF: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap, breaking at sentence boundaries when possible
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    text_len = len(text)
    
    if text_len < chunk_size:
        print("⚠ Document is smaller than chunk size, returning as single chunk")
        return [text]
    
    chunks = []
    start = 0
    chunk_count = 0
    
    while start < text_len:
        # Calculate end position
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence boundary if not at document end
        if end < text_len:
            # Search in last 30% of chunk for sentence endings
            search_start = end - int(chunk_size * 0.3)
            
            # Find last sentence ending
            best_break = -1
            for pattern in ['. ', '.\n', '? ', '!\n']:
                pos = text.rfind(pattern, search_start, end)
                if pos > best_break:
                    best_break = pos
            
            if best_break >= search_start:
                end = best_break + 1
        
        # Extract and store chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            chunk_count += 1
        
        # CRITICAL: Calculate next start position and ensure forward progress
        next_start = end - overlap
        
        # Ensure we always move forward to prevent infinite loop
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        
        start = next_start
        
        # Progress indicator
        if chunk_count % 50 == 0:
            print(f"Created {chunk_count} chunks...")
    
    print(f"✓ Created {len(chunks)} chunks")
    if chunks:
        avg_size = sum(len(c) for c in chunks) // len(chunks)
        print(f"Average chunk size: {avg_size} characters")
    
    return chunks


def get_chunk_statistics(chunks: List[str]) -> dict:
    """
    Calculate statistics about the chunks
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {}
    
    lengths = [len(c) for c in chunks]
    
    stats = {
        'num_chunks': len(chunks),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': sum(lengths) // len(lengths),
        'total_chars': sum(lengths)
    }
    
    return stats


def save_chunks(chunks: List[str], output_path: str = 'data/raw_chunks/'):
    """
    Save chunks to individual text files
    
    Args:
        chunks: List of text chunks
        output_path: Directory to save chunks
    """
    import os
    
    os.makedirs(output_path, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        filename = f"{output_path}chunk_{i:04d}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chunk)
    
    print(f"✓ Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    # Test the PDF processor
    print("=" * 60)
    print("PDF PROCESSOR TEST")
    print("=" * 60)
    
    # Load PDF
    pdf_path = "data/Uber_Annual_Report_2024.pdf"
    text = load_and_clean_pdf(pdf_path)
    
    print("\n" + "=" * 60)
    print("CHUNKING")
    print("=" * 60)
    
    # Create chunks
    chunks = chunk_text(text, chunk_size=1500, overlap=200)
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    # Get statistics
    stats = get_chunk_statistics(chunks)
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)
    
    # Show first chunk
    print(f"\nChunk 0 (first {200} chars):")
    print(chunks[0][:200] + "...")
    
    # Show middle chunk
    mid_idx = len(chunks) // 2
    print(f"\nChunk {mid_idx} (middle, first {200} chars):")
    print(chunks[mid_idx][:200] + "...")
    
    # Save chunks
    print("\n" + "=" * 60)
    print("SAVING CHUNKS")
    print("=" * 60)
    save_chunks(chunks)
    
    print("\n✓ PDF processing test complete!")
