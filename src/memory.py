"""ReasoningBank memory system implementation."""
import os
import json
import uuid
from typing import List, Optional
from pathlib import Path
import numpy as np
import faiss
from loguru import logger

from src.models import MemoryItem
from src.embeddings import EmbeddingProvider


class ReasoningBank:
    """
    ReasoningBank memory system with embedding-based retrieval.
    
    Components:
    - Storage: JSONL file for memory items
    - Index: FAISS vector index for similarity search
    - Retriever: Top-K cosine similarity retrieval
    """
    
    def __init__(
        self,
        bank_path: str,
        embedding_provider: EmbeddingProvider,
        dedup_threshold: float = 0.9
    ):
        self.bank_path = Path(bank_path)
        self.bank_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_provider = embedding_provider
        self.dedup_threshold = dedup_threshold
        
        self.memory_file = self.bank_path / "memories.jsonl"
        self.index_file = self.bank_path / "faiss.index"
        
        # Initialize storage
        self.memories: List[MemoryItem] = []
        self.index: Optional[faiss.Index] = None
        
        # Load existing memories if available
        self._load_memories()
    
    def _load_memories(self):
        """Load memories from disk."""
        if not self.memory_file.exists():
            logger.info("No existing memory bank found, starting fresh")
            return
        
        logger.info(f"Loading memories from {self.memory_file}")
        with open(self.memory_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    memory = MemoryItem(**data)
                    self.memories.append(memory)
        
        # Rebuild FAISS index
        if self.memories:
            self._rebuild_index()
        
        logger.info(f"Loaded {len(self.memories)} memories")
    
    def _rebuild_index(self):
        """Rebuild FAISS index from all memories."""
        embeddings = []
        for memory in self.memories:
            if memory.embedding is not None:
                embeddings.append(memory.embedding)
            else:
                # Generate embedding if missing
                text = self._memory_to_text(memory)
                emb = self.embedding_provider.embed([text])[0]
                memory.embedding = emb.tolist()
                embeddings.append(emb)
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            
            # Create FAISS index (inner product = cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            logger.info(f"Rebuilt FAISS index with {len(embeddings)} vectors")
    
    def _memory_to_text(self, memory: MemoryItem) -> str:
        """Convert memory item to text for embedding."""
        content_str = " ".join(memory.content)
        return f"{memory.title}. {memory.description}. {content_str}"
    
    def add_memory(self, memory: MemoryItem, check_duplicate: bool = True) -> bool:
        """
        Add a memory item to the bank.
        
        Returns:
            True if added, False if duplicate
        """
        # Generate embedding if not present
        if memory.embedding is None:
            text = self._memory_to_text(memory)
            embedding = self.embedding_provider.embed([text])[0]
            memory.embedding = embedding.tolist()
        
        # Check for duplicates
        if check_duplicate and self._is_duplicate(memory):
            logger.debug(f"Skipping duplicate memory: {memory.title}")
            return False
        
        # Add to storage
        self.memories.append(memory)
        
        # Append to file
        with open(self.memory_file, "a") as f:
            f.write(memory.model_dump_json() + "\n")
        
        # Update index
        if self.index is None:
            self._rebuild_index()
        else:
            embedding_array = np.array([memory.embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            self.index.add(embedding_array)
        
        logger.info(f"Added memory: {memory.title} (total: {len(self.memories)})")
        return True
    
    def _is_duplicate(self, memory: MemoryItem) -> bool:
        """Check if memory is duplicate using cosine similarity."""
        if not self.memories or self.index is None:
            return False
        
        # Search for similar memories
        query_embedding = np.array([memory.embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k=1)
        
        if len(distances) > 0 and distances[0][0] > self.dedup_threshold:
            return True
        
        return False
    
    def retrieve(self, query: str, k: int = 1, expected_answer: Optional[str] = None) -> List[MemoryItem]:
        """
        Retrieve top-K most similar memories for a query.

        Args:
            query: Query text (e.g., task description)
            k: Number of memories to retrieve
            expected_answer: Expected answer to filter out (prevents answer leakage)

        Returns:
            List of top-K memory items
        """
        if not self.memories or self.index is None:
            logger.warning("No memories available for retrieval")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed([query])[0]
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search for more candidates to allow filtering
        k_candidates = min(k * 3, len(self.memories))
        distances, indices = self.index.search(query_array, k_candidates)
        
        # Retrieve and filter memories
        generic_keywords = ["search", "navigate", "click", "scroll", "page", "explore", "browse"]
        retrieved = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):
                memory = self.memories[idx]

                # Answer leak protection: filter out memories containing expected answer
                if expected_answer:
                    memory_text = f"{memory.title} {memory.description} {memory.content}".lower()
                    if expected_answer.lower() in memory_text:
                        logger.debug(f"Filtered out memory with answer leak: {memory.title}")
                        continue

                # Check if memory is too generic
                title_lower = memory.title.lower()
                desc_lower = memory.description.lower()
                combined = title_lower + " " + desc_lower

                generic_count = sum(1 for kw in generic_keywords if kw in combined)

                # Prefer specific memories (max 2 generic keywords)
                # But don't filter ALL memories if we have nothing else
                if generic_count <= 2 or len(retrieved) < k // 2:
                    logger.debug(
                        f"Retrieved: {memory.title} (similarity: {distances[0][i]:.3f}, "
                        f"generic_keywords: {generic_count})"
                    )
                    retrieved.append(memory)
                else:
                    logger.debug(
                        f"Filtered out generic: {memory.title} (generic_keywords: {generic_count})"
                    )
                
                if len(retrieved) >= k:
                    break
        
        return retrieved[:k]
    
    def format_for_injection(self, memories: List[MemoryItem]) -> str:
        """
        Format retrieved memories for injection into agent prompt.
        
        Returns formatted string with memory content.
        """
        if not memories:
            return ""
        
        strategy_word = "strategies" if len(memories) > 1 else "strategy"
        parts = [
            f"## Relevant {strategy_word.capitalize()} from Similar Tasks",
            ""
        ]

        for i, memory in enumerate(memories, 1):
            parts.append(f"**{memory.title}**")
            if memory.description:
                parts.append(f"{memory.description}")
            parts.append("")
            for bullet in memory.content:
                parts.append(f"- {bullet}")
            parts.append("")
        
        return "\n".join(parts)
    
    def save_checkpoint(self):
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            logger.info(f"Saved FAISS index to {self.index_file}")
    
    def get_stats(self) -> dict:
        """Get memory bank statistics."""
        return {
            "total_memories": len(self.memories),
            "bank_path": str(self.bank_path),
            "index_size": self.index.ntotal if self.index else 0
        }
