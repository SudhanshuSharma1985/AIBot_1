import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_matches(query, text_embeddings, top_n=3):
    """
    Retrieve the top N best matches for a user query using cosine similarity.
    
    Args:
        query: The user query string
        text_embeddings: Array of embeddings with shape (n_samples, n_features)
        top_n: Number of top matches to return (default: 3)
    
    Returns:
        list: Top N matches with their similarity scores and indices
    """
    # Generate embedding for the query
    query_embedding = embedding.embed_query(query)
    
    # Ensure query_embedding is 2D for sklearn
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Compute cosine similarity between query and all embeddings at once
    # This is more efficient than computing similarities one by one
    similarities = cosine_similarity(query_embedding, text_embeddings)[0]
    
    # Get the indices of the top N matches using argpartition (faster than argsort for top-k)
    if top_n >= len(similarities):
        # If requesting more matches than available, use argsort
        top_indices = np.argsort(similarities)[::-1]
    else:
        # Use argpartition for better performance when top_n < total items
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        # Sort the top N indices by similarity score
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    # Retrieve the top N matches with their scores
    top_matches = [
        {
            'index': idx,
            'text': text_embeddings[idx][0] if isinstance(text_embeddings[idx], tuple) else idx,
            'similarity_score': similarities[idx]
        }
        for idx in top_indices
    ]
    
    return top_matches


# Alternative implementation using pure NumPy (if sklearn is not available)
def get_top_matches_numpy(query, text_embeddings, top_n=3):
    """
    Pure NumPy implementation for cosine similarity matching.
    """
    # Generate embedding for the query
    query_embedding = embedding.embed_query(query)
    query_embedding = np.array(query_embedding)
    
    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Extract embeddings from the array (assuming structure [[text, embedding], ...])
    if isinstance(text_embeddings[0], (list, tuple)) and len(text_embeddings[0]) == 2:
        embeddings_matrix = np.array([item[1] for item in text_embeddings])
        texts = [item[0] for item in text_embeddings]
    else:
        embeddings_matrix = text_embeddings
        texts = None
    
    # Normalize all embeddings at once
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized_embeddings = embeddings_matrix / norms
    
    # Compute cosine similarities using matrix multiplication
    similarities = np.dot(normalized_embeddings, query_norm)
    
    # Get top N indices efficiently
    if top_n >= len(similarities):
        top_indices = np.argsort(similarities)[::-1]
    else:
        # Use argpartition for O(n) complexity instead of O(n log n)
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    # Build result
    top_matches = []
    for idx in top_indices:
        match = {
            'index': idx,
            'similarity_score': similarities[idx]
        }
        if texts:
            match['text'] = texts[idx]
        top_matches.append(match)
    
    return top_matches


# Batch processing version for multiple queries
def get_top_matches_batch(queries, text_embeddings, top_n=3):
    """
    Process multiple queries at once for better efficiency.
    
    Args:
        queries: List of query strings
        text_embeddings: Array of embeddings
        top_n: Number of top matches per query
    
    Returns:
        dict: Dictionary mapping each query to its top matches
    """
    # Generate embeddings for all queries at once
    query_embeddings = np.array([embedding.embed_query(q) for q in queries])
    
    # Extract embedding matrix
    if isinstance(text_embeddings[0], (list, tuple)) and len(text_embeddings[0]) == 2:
        embeddings_matrix = np.array([item[1] for item in text_embeddings])
        texts = [item[0] for item in text_embeddings]
    else:
        embeddings_matrix = text_embeddings
        texts = None
    
    # Compute all similarities at once
    similarities = cosine_similarity(query_embeddings, embeddings_matrix)
    
    results = {}
    for i, query in enumerate(queries):
        # Get top matches for this query
        query_similarities = similarities[i]
        
        if top_n >= len(query_similarities):
            top_indices = np.argsort(query_similarities)[::-1]
        else:
            top_indices = np.argpartition(query_similarities, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(query_similarities[top_indices])[::-1]]
        
        matches = []
        for idx in top_indices:
            match = {
                'index': idx,
                'similarity_score': query_similarities[idx]
            }
            if texts:
                match['text'] = texts[idx]
            matches.append(match)
        
        results[query] = matches
    
    return results
