"""
The Librarian - RAG System for Operation Ledger-Mind
Implements hybrid search (Vector + BM25) using Weaviate
"""

import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
from typing import List, Dict
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


def connect_to_weaviate():
    """Connect to Weaviate Cloud instance"""
    load_environment()
    
    weaviate_url = os.getenv('WEAVIATE_URL')
    weaviate_key = os.getenv('WEAVIATE_API_KEY')
    
    if not weaviate_url or not weaviate_key:
        raise ValueError("Weaviate credentials not found in environment!")
    
    # Ensure URL has https://
    if not weaviate_url.startswith('http'):
        weaviate_url = f'https://{weaviate_url}'
    
    print(f"Connecting to Weaviate at {weaviate_url[:40]}...")
    
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_key),
            skip_init_checks=True  # Speed up connection
        )
        
        if client.is_ready():
            print("✓ Connected to Weaviate successfully")
            return client
        else:
            raise ConnectionError("Weaviate not ready")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("Note: Weaviate cluster may be sleeping. It will wake up on first request.")
        raise


def create_document_collection(client):
    """
    Create Weaviate collection for documents with hybrid search enabled
    """
    collection_name = "UberFinancialDocs"
    
    # Delete collection if it exists
    try:
        client.collections.delete(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection with hybrid search configuration
    client.collections.create(
        name=collection_name,
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
                description="Document chunk content"
            ),
            Property(
                name="chunk_id",
                data_type=DataType.INT,
                description="Chunk identifier"
            ),
            Property(
                name="source",
                data_type=DataType.TEXT,
                description="Source document"
            )
        ],
        # Enable vector search
        vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
        # Enable BM25 for keyword search
        inverted_index_config=Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2
        )
    )
    
    print(f"✓ Created collection: {collection_name}")
    return collection_name


def index_documents(client, collection_name: str, chunks: List[str], batch_size: int = 100):
    """
    Index document chunks into Weaviate with embeddings
    
    Args:
        client: Weaviate client
        collection_name: Name of the collection
        chunks: List of text chunks
        batch_size: Number of chunks to index per batch
    """
    print(f"\nIndexing {len(chunks)} chunks...")
    
    # Load sentence transformer for embeddings
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective
    
    # Get collection
    collection = client.collections.get(collection_name)
    
    # Batch index with embeddings
    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            # Generate embedding
            vector = embedder.encode(chunk).tolist()
            
            # Add to batch
            batch.add_object(
                properties={
                    "content": chunk,
                    "chunk_id": i,
                    "source": "Uber_Annual_Report_2024.pdf"
                },
                vector=vector
            )
            
            if (i + 1) % 50 == 0:
                print(f"Indexed {i + 1}/{len(chunks)} chunks...")
    
    print(f"✓ Indexed {len(chunks)} chunks successfully")


def hybrid_search(
    client,
    collection_name: str,
    query: str,
    embedder: SentenceTransformer,
    limit: int = 5,
    alpha: float = 0.5
) -> List[Dict]:
    """
    Perform hybrid search (Vector + BM25)
    
    Args:
        client: Weaviate client
        collection_name: Name of collection
        query: Search query
        embedder: Sentence transformer model
        limit: Number of results
        alpha: Balance between vector (1.0) and keyword (0.0) search
        
    Returns:
        List of search results with content and scores
    """
    # Generate query embedding
    query_vector = embedder.encode(query).tolist()
    
    # Get collection
    collection = client.collections.get(collection_name)
    
    # Hybrid search
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,  # 0.5 = balanced, 1.0 = vector only, 0.0 = keyword only
        limit=limit,
        return_metadata=['score']
    )
    
    # Format results
    results = []
    for obj in response.objects:
        results.append({
            "content": obj.properties["content"],
            "chunk_id": obj.properties["chunk_id"],
            "score": obj.metadata.score,
            "source": obj.properties["source"]
        })
    
    return results


def query_rag(query: str, client, collection_name: str, embedder: SentenceTransformer) -> str:
    """
    Query the RAG system and generate answer
    
    Args:
        query: User question
        client: Weaviate client
        collection_name: Collection name
        embedder: Embedding model
        
    Returns:
        Generated answer
    """
    # Get relevant chunks
    results = hybrid_search(client, collection_name, query, embedder, limit=3, alpha=0.7)
    
    # Combine context
    context = "\n\n".join([r["content"][:500] for r in results])
    
    # For now, return context (we'll add LLM generation later)
    return f"Based on the retrieved context:\n\n{context}\n\n[Note: LLM answer generation will be added in next step]"


if __name__ == "__main__":
    print("="*60)
    print("THE LIBRARIAN - RAG SYSTEM SETUP")
    print("="*60)
    
    client = None
    try:
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        # Create collection
        collection_name = create_document_collection(client)
        
        # Load document chunks
        print("\n" + "="*60)
        print("LOADING DOCUMENT CHUNKS")
        print("="*60)
        
        import glob
        chunk_files = sorted(glob.glob('data/raw_chunks/chunk_*.txt'))
        
        chunks = []
        for file_path in chunk_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks.append(f.read())
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Index documents
        print("\n" + "="*60)
        print("INDEXING DOCUMENTS")
        print("="*60)
        
        index_documents(client, collection_name, chunks)
        
        # Test search
        print("\n" + "="*60)
        print("TESTING HYBRID SEARCH")
        print("="*60)
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_queries = [
            "What was Uber's revenue in 2024?",
            "What are the main risk factors for Uber?",
            "How does Uber describe its market position?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = hybrid_search(client, collection_name, query, embedder, limit=2)
            print(f"Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n  Result {i} (score: {r['score']:.3f}):")
                print(f"  {r['content'][:150]}...")
        
        print("\n" + "="*60)
        print("✓ RAG System Setup Complete!")
        print("="*60)
    
    finally:
        # Always close connection
        if client is not None:
            try:
                client.close()
                print("\n✓ Connection closed cleanly")
            except:
                pass
