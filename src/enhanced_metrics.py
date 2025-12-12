#!/usr/bin/env python3
"""
Enhanced Metrics for ReasoningBank Evaluation

This module provides finer-grained metrics including:
- Retrieval quality metrics
- Memory content analysis
- Memory quantity vs performance analysis
- Task-specific metrics
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from loguru import logger
import pandas as pd


class RetrievalQualityMetrics:
    """Metrics for evaluating memory retrieval quality."""

    def __init__(self):
        self.retrieval_logs = []

    def log_retrieval(self, task_id: str, query: str, retrieved_memories: List[Dict],
                      success: bool, task_category: str = None):
        """Log a retrieval event."""
        self.retrieval_logs.append({
            "task_id": task_id,
            "query": query,
            "num_retrieved": len(retrieved_memories),
            "memory_ids": [m.get("id") for m in retrieved_memories],
            "similarity_scores": [m.get("similarity", 0) for m in retrieved_memories],
            "success": success,
            "task_category": task_category
        })

    def compute_retrieval_precision(self) -> Dict[str, float]:
        """
        Compute retrieval precision: % of retrieved memories that led to successful tasks.
        """
        if not self.retrieval_logs:
            return {"overall": 0.0}

        # Overall precision
        successful_retrievals = sum(1 for log in self.retrieval_logs if log["success"] and log["num_retrieved"] > 0)
        total_retrievals = sum(1 for log in self.retrieval_logs if log["num_retrieved"] > 0)

        results = {
            "overall": successful_retrievals / total_retrievals if total_retrievals > 0 else 0.0,
            "total_retrievals": total_retrievals,
            "successful_retrievals": successful_retrievals
        }

        # Per-category precision
        categories = defaultdict(lambda: {"success": 0, "total": 0})
        for log in self.retrieval_logs:
            if log["task_category"] and log["num_retrieved"] > 0:
                categories[log["task_category"]]["total"] += 1
                if log["success"]:
                    categories[log["task_category"]]["success"] += 1

        for cat, stats in categories.items():
            results[f"precision_{cat}"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0

        return results

    def compute_retrieval_coverage(self) -> Dict[str, float]:
        """
        Compute retrieval coverage: % of tasks that successfully retrieved memories.
        """
        if not self.retrieval_logs:
            return {"coverage": 0.0}

        tasks_with_retrieval = sum(1 for log in self.retrieval_logs if log["num_retrieved"] > 0)
        total_tasks = len(self.retrieval_logs)

        return {
            "coverage": tasks_with_retrieval / total_tasks if total_tasks > 0 else 0.0,
            "tasks_with_retrieval": tasks_with_retrieval,
            "total_tasks": total_tasks
        }

    def compute_similarity_statistics(self) -> Dict[str, float]:
        """Compute statistics about similarity scores."""
        all_scores = []
        for log in self.retrieval_logs:
            all_scores.extend(log["similarity_scores"])

        if not all_scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "min": np.min(all_scores),
            "max": np.max(all_scores),
            "median": np.median(all_scores),
            "q25": np.percentile(all_scores, 25),
            "q75": np.percentile(all_scores, 75)
        }

    def get_all_metrics(self) -> Dict:
        """Get all retrieval quality metrics."""
        return {
            "precision": self.compute_retrieval_precision(),
            "coverage": self.compute_retrieval_coverage(),
            "similarity_stats": self.compute_similarity_statistics()
        }


class MemoryContentAnalyzer:
    """Analyze memory bank content and quality."""

    def __init__(self, memory_bank_path: str):
        self.memory_bank_path = Path(memory_bank_path)
        self.memories = self._load_memories()

    def _load_memories(self) -> List[Dict]:
        """Load all memories from the memory bank."""
        memories = []
        if not self.memory_bank_path.exists():
            return memories

        # Load from JSON files in memory bank
        for file in self.memory_bank_path.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    memories.extend(data)
                else:
                    memories.append(data)

        return memories

    def analyze_memory_quantity(self) -> Dict[str, int]:
        """Analyze quantity metrics of memory bank."""
        total_memories = len(self.memories)

        # Count by source (success vs failure)
        success_count = sum(1 for m in self.memories if m.get("source") == "success")
        failure_count = sum(1 for m in self.memories if m.get("source") == "failure")

        # Count by category if available
        categories = defaultdict(int)
        for m in self.memories:
            cat = m.get("category", "unknown")
            categories[cat] += 1

        return {
            "total_memories": total_memories,
            "success_memories": success_count,
            "failure_memories": failure_count,
            "categories": dict(categories),
            "success_ratio": success_count / total_memories if total_memories > 0 else 0.0
        }

    def analyze_memory_quality(self) -> Dict[str, float]:
        """Analyze quality metrics of memories."""
        if not self.memories:
            return {"avg_content_length": 0, "avg_title_length": 0}

        # Analyze content length
        content_lengths = [len(m.get("content", "")) for m in self.memories]
        title_lengths = [len(m.get("title", "")) for m in self.memories]

        # Count unique vs duplicate strategies
        unique_titles = len(set(m.get("title", "") for m in self.memories))

        return {
            "avg_content_length": np.mean(content_lengths),
            "std_content_length": np.std(content_lengths),
            "avg_title_length": np.mean(title_lengths),
            "unique_strategies": unique_titles,
            "uniqueness_ratio": unique_titles / len(self.memories) if len(self.memories) > 0 else 0.0
        }

    def analyze_memory_evolution(self, logs_dir: str) -> Dict:
        """Analyze how memory bank evolves over time."""
        # This would track memory growth over task sequence
        # For now, return basic stats
        return {
            "final_size": len(self.memories),
            "growth_rate": "N/A"  # Would need temporal data
        }

    def get_all_metrics(self) -> Dict:
        """Get all memory content metrics."""
        return {
            "quantity": self.analyze_memory_quantity(),
            "quality": self.analyze_memory_quality(),
            "evolution": self.analyze_memory_evolution("")
        }


class MemoryQuantityExperiment:
    """Experiment with different memory quantities."""

    @staticmethod
    def analyze_quantity_vs_performance(results_dir: str) -> pd.DataFrame:
        """
        Analyze relationship between memory quantity and performance.

        Args:
            results_dir: Directory containing experiment results

        Returns:
            DataFrame with memory quantity vs performance metrics
        """
        results_path = Path(results_dir)
        data = []

        # Load results from different memory configurations
        for result_file in results_path.glob("**/results.json"):
            with open(result_file, "r") as f:
                result = json.load(f)

            # Extract memory count and performance
            memory_count = result.get("memory_count", 0)
            success_rate = result.get("success_rate", 0)
            avg_steps = result.get("avg_steps", 0)

            data.append({
                "memory_count": memory_count,
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "config": result_file.parent.name
            })

        df = pd.DataFrame(data)
        return df.sort_values("memory_count")


class TaskSpecificMetrics:
    """Compute task-specific performance metrics."""

    @staticmethod
    def compute_per_task_metrics(results: List[Dict]) -> pd.DataFrame:
        """Compute detailed metrics for each task."""
        rows = []

        for result in results:
            row = {
                "task_id": result.get("task_id"),
                "success": result.get("success", False),
                "steps": result.get("steps", 0),
                "time_seconds": result.get("time_seconds", 0),
                "memory_used": result.get("memory_used", False),
                "num_memories_retrieved": result.get("num_memories_retrieved", 0),
                "category": result.get("category", "unknown"),
                "difficulty": result.get("difficulty", "unknown")
            }
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def compute_category_breakdown(df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute metrics broken down by task category."""
        if "category" not in df.columns:
            return {}

        breakdown = {}
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            breakdown[category] = {
                "count": len(cat_df),
                "success_rate": cat_df["success"].mean(),
                "avg_steps": cat_df["steps"].mean(),
                "avg_time": cat_df["time_seconds"].mean()
            }

        return breakdown

    @staticmethod
    def compute_difficulty_breakdown(df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute metrics broken down by task difficulty."""
        if "difficulty" not in df.columns:
            return {}

        breakdown = {}
        for difficulty in df["difficulty"].unique():
            diff_df = df[df["difficulty"] == difficulty]
            breakdown[difficulty] = {
                "count": len(diff_df),
                "success_rate": diff_df["success"].mean(),
                "avg_steps": diff_df["steps"].mean(),
                "avg_time": diff_df["time_seconds"].mean()
            }

        return breakdown


def compute_comprehensive_metrics(
    results_path: str,
    memory_bank_path: str,
    retrieval_logs_path: Optional[str] = None
) -> Dict:
    """
    Compute all enhanced metrics for an experiment.

    Args:
        results_path: Path to results JSON file
        memory_bank_path: Path to memory bank directory
        retrieval_logs_path: Optional path to retrieval logs

    Returns:
        Dictionary containing all metrics
    """
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)

    # Initialize analyzers
    memory_analyzer = MemoryContentAnalyzer(memory_bank_path)
    retrieval_metrics = RetrievalQualityMetrics()

    # Load retrieval logs if available
    if retrieval_logs_path and Path(retrieval_logs_path).exists():
        with open(retrieval_logs_path, "r") as f:
            retrieval_logs = json.load(f)
        for log in retrieval_logs:
            retrieval_metrics.retrieval_logs.append(log)

    # Compute all metrics
    metrics = {
        "basic_metrics": {
            "success_rate": results.get("success_rate", 0),
            "avg_steps": results.get("avg_steps", 0),
            "total_tasks": results.get("total_tasks", 0)
        },
        "retrieval_quality": retrieval_metrics.get_all_metrics(),
        "memory_content": memory_analyzer.get_all_metrics(),
        "task_specific": {}
    }

    # Compute task-specific metrics if detailed results available
    if "task_results" in results:
        df = TaskSpecificMetrics.compute_per_task_metrics(results["task_results"])
        metrics["task_specific"] = {
            "category_breakdown": TaskSpecificMetrics.compute_category_breakdown(df),
            "difficulty_breakdown": TaskSpecificMetrics.compute_difficulty_breakdown(df)
        }

    return metrics


if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Metrics Module")
    logger.info("This module provides comprehensive evaluation metrics for ReasoningBank")
