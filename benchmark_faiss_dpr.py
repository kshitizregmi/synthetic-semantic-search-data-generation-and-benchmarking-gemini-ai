import json
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import faiss
import numpy as np

# Load models
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

def encode_docs(docs):
    """Encode all documents in batch"""
    inputs = context_tokenizer(docs, padding=True, truncation=True, max_length=512, return_tensors='pt')
    embeddings = context_encoder(**inputs).pooler_output.detach().numpy()
    return embeddings

def encode_query(query):
    """Encode single query"""
    inputs = question_tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors='pt')
    embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    return embedding

def evaluate_recall(data_file):
    # Load and process all documents
    all_docs = []
    query_to_relevant = {}
    
    # Extract all unique documents and create mapping
    for item in data_file:
        query = item['query']
        relevant_docs = item['relevant_docs']
        query_to_relevant[query] = set(relevant_docs)  # Store relevant docs for each query
        all_docs.extend(relevant_docs)
    
    # Remove duplicates while preserving order
    unique_docs = list(dict.fromkeys(all_docs))
    doc_to_idx = {doc: idx for idx, doc in enumerate(unique_docs)}
    
    # Create FAISS index
    doc_embeddings = encode_docs(unique_docs)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    # Evaluate each query
    total_recall = 0
    num_queries = len(data_file)
    
    for item in data_file:
        query = item['query']
        query_embedding = encode_query(query)
        
        # Get top 3 retrieved documents
        _, indices = index.search(query_embedding, k=3)
        retrieved_docs = {unique_docs[idx] for idx in indices[0]}
        
        # Get relevant documents for this query
        relevant_docs = query_to_relevant[query]
        
        # Calculate recall for this query
        # For each query, we have exactly 3 relevant docs
        num_relevant_retrieved = len(retrieved_docs.intersection(relevant_docs))
        recall = num_relevant_retrieved / 3
        total_recall += recall
    
    avg_recall = total_recall / num_queries
    return avg_recall

if __name__ == "__main__":
    # Load the data
    with open('synthetic_data.json', 'r') as file:
        data = json.load(file)
    
    # Calculate Recall@3
    recall = evaluate_recall(data)
    print(f"Recall@3: {recall:.4f}")
    print(f"Number of queries: {len(data)}")