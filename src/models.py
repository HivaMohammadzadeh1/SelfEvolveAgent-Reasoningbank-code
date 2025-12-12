"""Data models for ReasoningBank implementation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryItem(BaseModel):
    """
    Structured memory item from ReasoningBank paper.
    
    Core components (shown to agent):
        - title: Concise identifier of strategy/pattern
        - description: One-sentence summary
        - content: Distilled reasoning steps (1-3 sentences)
    
    System fields (not shown to agent):
        - id: Unique identifier for tracking
        - provenance: Metadata (task_id, success, timestamp)
        - embedding: Vector for similarity retrieval
    """
    # Core memory components (from paper's memory schema)
    title: str = Field(description="Concise identifier summarizing the core strategy or reasoning pattern")
    description: str = Field(description="Brief one-sentence summary of the memory item")
    content: List[str] = Field(description="Distilled reasoning steps, decision rationales, or operational insights")
    
    # System fields (for storage/retrieval, not shown to agent)
    id: str = Field(description="Unique identifier for memory item")
    provenance: Dict[str, Any] = Field(description="Source metadata (task_id, success, timestamp)")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector for similarity search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "mem_001",
                "title": "Prioritize account sections over global search",
                "description": "When query is user-history related, check account/orders first",
                "content": [
                    "Look for 'My Account' or 'Order History' first",
                    "Prefer 'View all' to avoid infinite scroll traps",
                    "If pagination exists, traverse oldest→newest deterministically"
                ],
                "rationale": "Account sections are more reliable than search",
                "provenance": {
                    "task_id": "task_123",
                    "success": True,
                    "timestamp": "2025-10-07T10:00:00Z"
                }
            }
        }


class Action(BaseModel):
    """Agent action in a trajectory."""
    
    tool: str = Field(description="Tool name (e.g., navigate, click, type)")
    args: Dict[str, Any] = Field(description="Tool arguments")
    timestamp: Optional[str] = None


class Trajectory(BaseModel):
    """Complete agent trajectory for a task."""

    task_id: str
    task_description: Optional[str] = None  # Original task question
    reference_answer: Optional[str] = None  # Expected/correct answer
    seed: int = 42
    success: bool = False
    steps: int = 0
    final_answer: Optional[str] = None
    actions: List[Action] = Field(default_factory=list)
    predicted_actions: List[Dict[str, Any]] = Field(default_factory=list)  # For Mind2Web evaluation
    ground_truth_actions: List[Dict[str, Any]] = Field(default_factory=list)  # Ground truth actions for Mind2Web
    observations: List[str] = Field(default_factory=list)
    thoughts: List[str] = Field(default_factory=list)
    tokens: Dict[str, int] = Field(default_factory=lambda: {"input": 0, "output": 0})
    walltime: float = 0.0
    backbone: str = "gpt-4"
    config: Dict[str, Any] = Field(default_factory=dict)
    retrieved_memories: List[MemoryItem] = Field(default_factory=list)
    

class TaskResult(BaseModel):
    """Result of evaluating a single task."""
    
    task_id: str
    subset: str  # shopping, admin, gitlab, reddit, multi
    success: bool
    steps: int
    tokens_input: int
    tokens_output: int
    walltime: float
    seed: int
    error: Optional[str] = None
    trajectory_path: Optional[str] = None
    agent_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    evaluation_explanation: Optional[str] = None


class EvaluationResults(BaseModel):
    """Aggregated evaluation results."""

    mode: str  # "no_memory" or "reasoningbank"
    subset: str
    success_rate: float
    avg_steps: float
    total_tasks: int
    successful_tasks: int
    total_tokens: int
    total_walltime: float
    task_results: List[TaskResult] = Field(default_factory=list)
    mind2web_metrics: Optional[Dict[str, float]] = None  # Mind2Web-specific metrics (EA, AF1, SSR, SR)
