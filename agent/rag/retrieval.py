import os
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    """
    A simple TF-IDF based retriever for fetching document chunks.
    """
    def __init__(self, doc_path: str = "docs"):
        self.doc_path = doc_path
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self._load_and_chunk_docs()

    def _load_and_chunk_docs(self):
        """Loads documents and splits them into paragraph-level chunks."""
        for filename in os.listdir(self.doc_path):
            if filename.endswith(".md"):
                filepath = os.path.join(self.doc_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split by paragraphs (one or more newlines)
                    paragraphs = re.split(r'\n\s*\n', content)
                    for i, chunk in enumerate(paragraphs):
                        if chunk.strip():
                            self.chunks.append(chunk)
                            # Create a unique chunk ID: filename::chunkX
                            source_id = f"{os.path.splitext(filename)[0]}::chunk{i}"
                            self.chunk_sources.append(source_id)
        
        if not self.chunks:
            raise ValueError("No documents found or processed. Check the 'docs' directory.")
            
        # Fit the vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print(f"Retriever initialized with {len(self.chunks)} chunks from {len(os.listdir(self.doc_path))} files.")

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[str], List[str]]:
        """
        Retrieves the top-k most relevant document chunks for a given query.

        Returns:
            A tuple containing:
            - A list of the content of the top-k chunks.
            - A list of the source IDs of the top-k chunks.
        """
        if self.tfidf_matrix is None:
            return [], []
            
        # Vectorize the query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices. Use argpartition for efficiency.
        # This gets the indices of the k largest values.
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        
        # Sort these k indices by similarity score
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        # Get the corresponding chunks and sources
        retrieved_contents = [self.chunks[i] for i in top_k_indices]
        retrieved_citations = [self.chunk_sources[i] for i in top_k_indices]
        
        return retrieved_contents, retrieved_citations