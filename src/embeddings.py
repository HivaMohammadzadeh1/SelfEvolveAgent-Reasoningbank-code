"""Embedding generation for memory retrieval."""
import os
import time
from typing import List
import numpy as np
from abc import ABC, abstractmethod
import hashlib

# Optional imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

# Don't import sentence_transformers at module level - it's slow!
# Import it only when actually needed in the class __init__
SENTENCE_TRANSFORMERS_AVAILABLE = None  # Will check on first use


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-large"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Install with: pip install openai")
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._dimension = 3072 if "large" in model else 1536
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini embedding provider with rate limiting."""

    def __init__(self, model: str = "models/embedding-001", rate_limit_delay: float = 2.0):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
        self.model = model
        # Rate limiting to avoid hitting API limits
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # Gemini embeddings have dimension 768
        self._dimension = 768

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Google Gemini API with rate limiting."""
        embeddings = []
        for text in texts:
            # Rate limiting: ensure minimum delay between API calls
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                time.sleep(sleep_time)

            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

            # Update last call time
            self.last_call_time = time.time()

        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class TogetherAIEmbedding(EmbeddingProvider):
    """TogetherAI embedding provider."""

    def __init__(self, model: str = "togethercomputer/m2-bert-80M-32k-retrieval", rate_limit_delay: float = 0.1):
        if not TOGETHER_AVAILABLE:
            raise ImportError("together package not installed. Install with: pip install together")
        self.model = model
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        # Rate limiting to avoid hitting API limits
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0

        # Model dimensions - m2-bert-80M outputs 768-dim embeddings
        # Adjust based on the specific model
        if "m2-bert" in model:
            self._dimension = 768
        else:
            # Default for most embedding models
            self._dimension = 768

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using TogetherAI API with rate limiting."""
        embeddings = []
        for text in texts:
            # Rate limiting: ensure minimum delay between API calls
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                time.sleep(sleep_time)

            # Call TogetherAI embeddings API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)

            # Update last call time
            self.last_call_time = time.time()

        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""
    
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import - only import when actually creating an instance
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class SimpleHashEmbedding(EmbeddingProvider):
    """Simple hash-based embedding (fallback when no other provider available)."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate simple hash-based embeddings."""
        embeddings = []
        for text in texts:
            # Create multiple hashes to fill dimension
            embedding = []
            for i in range(0, self._dimension, 32):
                # Hash with different seeds
                hash_input = f"{text}_{i}".encode('utf-8')
                hash_obj = hashlib.sha256(hash_input)
                hash_bytes = hash_obj.digest()
                # Convert to floats
                vals = [float(b) / 255.0 - 0.5 for b in hash_bytes]
                embedding.extend(vals[:min(32, self._dimension - i)])
            
            embeddings.append(embedding[:self._dimension])
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_provider(provider: str, model: str, rate_limit_delay: float = 2.0) -> EmbeddingProvider:
    """
    Factory function to create embedding provider.

    Args:
        provider: Embedding provider (openai, google, together, sentence_transformers, simple)
        model: Model name
        rate_limit_delay: Minimum delay in seconds between API calls (for Google and TogetherAI)
    """

    if provider == "openai":
        if not OPENAI_AVAILABLE:
            print("Warning: OpenAI not available, using simple hash embeddings")
            return SimpleHashEmbedding()
        return OpenAIEmbedding(model)
    elif provider == "google":
        if not GOOGLE_AVAILABLE:
            print("Warning: Google Generative AI not available, using simple hash embeddings")
            return SimpleHashEmbedding()
        return GeminiEmbedding(model, rate_limit_delay=rate_limit_delay)
    elif provider == "together":
        if not TOGETHER_AVAILABLE:
            print("Warning: TogetherAI not available, using simple hash embeddings")
            return SimpleHashEmbedding()
        return TogetherAIEmbedding(model, rate_limit_delay=rate_limit_delay)
    elif provider == "sentence_transformers":
        # Try to create SentenceTransformerEmbedding
        # It will check availability when instantiated
        try:
            return SentenceTransformerEmbedding(model)
        except ImportError:
            print("Warning: sentence-transformers not available, using simple hash embeddings")
            return SimpleHashEmbedding()
    elif provider == "simple":
        return SimpleHashEmbedding()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
