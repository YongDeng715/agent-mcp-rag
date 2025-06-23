from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingRetriever:
    """Vector retrieval implementation for RAG"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        
    async def embed_document(self, document: str):
        """Generate and store document embedding"""
        embedding = await self._get_embedding(document)
        self.documents.append(document)
        self.embeddings.append(embedding)
        
    async def embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        return await self._get_embedding(query)
        
    async def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top_k most relevant documents"""
        query_embedding = await self.embed_query(query)
        
        # Calculate cosine similarities
        similarities = [
            self._cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in self.embeddings
        ]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
        
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using model"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0].numpy()
        return embedding
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))