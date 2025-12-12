#!/usr/bin/env python3
"""
Task Analyzer for ReasoningBank

Classifies and analyzes tasks by:
- Difficulty (easy, medium, hard)
- Category (information seeking, navigation, content modification)
- Complexity metrics
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path
from loguru import logger
from collections import defaultdict
import numpy as np


class TaskDifficultyClassifier:
    """Classify tasks into difficulty levels based on multiple criteria."""

    # Difficulty heuristics based on query characteristics
    EASY_KEYWORDS = [
        "what is", "tell me", "show me", "list", "find", "get",
        "how many", "when", "where", "who"
    ]

    HARD_KEYWORDS = [
        "compare", "analyze", "evaluate", "calculate", "aggregate",
        "filter and", "if", "unless", "except", "between", "among",
        "top", "best", "worst", "most", "least"
    ]

    def __init__(self):
        self.difficulty_cache = {}

    def classify_by_query_complexity(self, query: str) -> str:
        """Classify difficulty based on query text complexity."""
        query_lower = query.lower()

        # Count complexity indicators
        word_count = len(query.split())
        has_easy_keywords = any(kw in query_lower for kw in self.EASY_KEYWORDS)
        has_hard_keywords = any(kw in query_lower for kw in self.HARD_KEYWORDS)
        has_multiple_conditions = query_lower.count(" and ") > 1 or query_lower.count(" or ") > 1
        has_negation = " not " in query_lower or "n't" in query_lower

        # Scoring system
        difficulty_score = 0

        # Length contributes to difficulty
        if word_count > 20:
            difficulty_score += 2
        elif word_count > 12:
            difficulty_score += 1

        # Keywords
        if has_easy_keywords and not has_hard_keywords:
            difficulty_score -= 1
        if has_hard_keywords:
            difficulty_score += 2

        # Logical complexity
        if has_multiple_conditions:
            difficulty_score += 2
        if has_negation:
            difficulty_score += 1

        # Classify based on score
        if difficulty_score <= 0:
            return "easy"
        elif difficulty_score <= 3:
            return "medium"
        else:
            return "hard"

    def classify_by_execution_metrics(
        self,
        baseline_steps: int,
        baseline_success: bool,
        avg_steps: float
    ) -> str:
        """
        Classify difficulty based on execution metrics from baseline (no memory).

        Args:
            baseline_steps: Number of steps taken in baseline
            baseline_success: Whether baseline succeeded
            avg_steps: Average steps across all tasks

        Returns:
            Difficulty classification: "easy", "medium", or "hard"
        """
        # Tasks that fail in baseline are hard
        if not baseline_success:
            return "hard"

        # Tasks that take many more steps than average are harder
        if baseline_steps > avg_steps * 1.5:
            return "hard"
        elif baseline_steps > avg_steps * 1.2:
            return "medium"
        else:
            return "easy"

    def classify_task(
        self,
        task_id: str,
        query: str,
        baseline_steps: int = None,
        baseline_success: bool = None,
        avg_steps: float = None
    ) -> Dict[str, str]:
        """
        Comprehensive task difficulty classification.

        Returns:
            Dictionary with multiple difficulty assessments
        """
        if task_id in self.difficulty_cache:
            return self.difficulty_cache[task_id]

        result = {
            "task_id": task_id,
            "query_based": self.classify_by_query_complexity(query)
        }

        # Add execution-based classification if metrics available
        if baseline_steps is not None and baseline_success is not None and avg_steps is not None:
            result["execution_based"] = self.classify_by_execution_metrics(
                baseline_steps, baseline_success, avg_steps
            )

            # Combined difficulty (take the harder of the two)
            difficulties = ["easy", "medium", "hard"]
            query_idx = difficulties.index(result["query_based"])
            exec_idx = difficulties.index(result["execution_based"])
            combined_idx = max(query_idx, exec_idx)
            result["combined"] = difficulties[combined_idx]
        else:
            result["combined"] = result["query_based"]

        self.difficulty_cache[task_id] = result
        return result


class TaskCategoryClassifier:
    """Classify tasks into categories based on intent."""

    CATEGORIES = {
        "information_seeking": {
            "keywords": [
                "what", "tell me", "show me", "find", "get", "list",
                "how many", "count", "sum", "total", "number of",
                "which", "who", "when", "where", "why"
            ],
            "description": "User wants to retrieve information"
        },
        "navigation": {
            "keywords": [
                "go to", "navigate", "open", "visit", "access",
                "click on", "select", "choose", "view"
            ],
            "description": "User wants to navigate to a page"
        },
        "content_modification": {
            "keywords": [
                "change", "update", "modify", "edit", "set",
                "add", "create", "delete", "remove", "post",
                "submit", "save", "configure"
            ],
            "description": "User wants to modify content"
        },
        "comparison": {
            "keywords": [
                "compare", "difference", "versus", "vs", "better",
                "worse", "more than", "less than", "between"
            ],
            "description": "User wants to compare items"
        },
        "aggregation": {
            "keywords": [
                "total", "sum", "average", "count", "aggregate",
                "all", "every", "each", "list of", "top", "bottom"
            ],
            "description": "User wants aggregated information"
        },
        "filtering": {
            "keywords": [
                "filter", "where", "that", "which", "with",
                "having", "matching", "containing", "ending", "starting"
            ],
            "description": "User wants filtered results"
        }
    }

    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify a task into one or more categories.

        Returns:
            Dictionary with primary category and confidence scores
        """
        query_lower = query.lower()

        # Score each category
        scores = {}
        for category, config in self.CATEGORIES.items():
            keywords = config["keywords"]
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[category] = score

        # Get primary category (highest score)
        if max(scores.values()) == 0:
            primary = "uncategorized"
            confidence = 0.0
        else:
            primary = max(scores, key=scores.get)
            total_matches = sum(scores.values())
            confidence = scores[primary] / total_matches if total_matches > 0 else 0.0

        # Get secondary categories (with positive scores)
        secondary = [cat for cat, score in scores.items() if score > 0 and cat != primary]

        return {
            "primary_category": primary,
            "confidence": confidence,
            "secondary_categories": secondary,
            "all_scores": scores
        }


class SubsetDistributionAnalyzer:
    """Analyze distribution of tasks within each subset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.difficulty_classifier = TaskDifficultyClassifier()
        self.category_classifier = TaskCategoryClassifier()

    def analyze_subset(
        self,
        subset_name: str,
        baseline_results: Dict = None
    ) -> Dict:
        """
        Analyze distribution of a specific subset.

        Args:
            subset_name: Name of subset (e.g., "shopping", "admin")
            baseline_results: Optional baseline results for execution-based metrics

        Returns:
            Dictionary with distribution statistics
        """
        # Load tasks for this subset
        tasks = self._load_subset_tasks(subset_name)

        if not tasks:
            logger.warning(f"No tasks found for subset: {subset_name}")
            return {}

        # Compute average steps from baseline if available
        avg_steps = None
        if baseline_results:
            steps = [r["steps"] for r in baseline_results.get("task_results", [])]
            avg_steps = np.mean(steps) if steps else None

        # Classify all tasks
        difficulties = defaultdict(int)
        categories = defaultdict(int)
        task_classifications = []

        for task in tasks:
            task_id = task.get("task_id")
            query = task.get("intent", "")

            # Get baseline metrics for this task if available
            baseline_steps = None
            baseline_success = None
            if baseline_results:
                task_result = next(
                    (r for r in baseline_results.get("task_results", [])
                     if r.get("task_id") == task_id),
                    None
                )
                if task_result:
                    baseline_steps = task_result.get("steps")
                    baseline_success = task_result.get("success")

            # Classify difficulty
            diff_result = self.difficulty_classifier.classify_task(
                task_id, query, baseline_steps, baseline_success, avg_steps
            )
            difficulty = diff_result.get("combined", "unknown")
            difficulties[difficulty] += 1

            # Classify category
            cat_result = self.category_classifier.classify(query)
            category = cat_result["primary_category"]
            categories[category] += 1

            task_classifications.append({
                "task_id": task_id,
                "difficulty": difficulty,
                "category": category,
                "query": query
            })

        # Compute distributions
        total_tasks = len(tasks)

        return {
            "subset": subset_name,
            "total_tasks": total_tasks,
            "difficulty_distribution": {
                difficulty: count / total_tasks
                for difficulty, count in difficulties.items()
            },
            "difficulty_counts": dict(difficulties),
            "category_distribution": {
                category: count / total_tasks
                for category, count in categories.items()
            },
            "category_counts": dict(categories),
            "task_classifications": task_classifications
        }

    def _load_subset_tasks(self, subset_name: str) -> List[Dict]:
        """Load tasks for a specific subset."""
        # Try multiple possible paths
        possible_paths = [
            self.dataset_path / f"test_{subset_name}.json",
            self.dataset_path / f"{subset_name}.json",
            self.dataset_path / "test.json"
        ]

        for path in possible_paths:
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)

                # Filter by subset if loading from combined file
                if isinstance(data, list):
                    # Check if subset info is in task objects
                    subset_tasks = [
                        task for task in data
                        if task.get("sites", [subset_name])[0] == subset_name
                    ]
                    if subset_tasks:
                        return subset_tasks
                    return data  # Return all if no subset field
                return []

        logger.warning(f"Could not find task file for subset: {subset_name}")
        return []

    def analyze_all_subsets(
        self,
        subsets: List[str],
        baseline_results_dir: str = None
    ) -> Dict[str, Dict]:
        """
        Analyze distribution for all subsets.

        Args:
            subsets: List of subset names
            baseline_results_dir: Directory with baseline results

        Returns:
            Dictionary mapping subset names to their distributions
        """
        results = {}

        for subset in subsets:
            logger.info(f"Analyzing subset: {subset}")

            # Load baseline results if available
            baseline_results = None
            if baseline_results_dir:
                baseline_file = Path(baseline_results_dir) / f"no_memory_{subset}.json"
                if baseline_file.exists():
                    with open(baseline_file, "r") as f:
                        baseline_results = json.load(f)

            # Analyze subset
            results[subset] = self.analyze_subset(subset, baseline_results)

        return results


def generate_distribution_report(
    distributions: Dict[str, Dict],
    output_path: str = "distribution_report.md"
):
    """Generate a markdown report of task distributions."""
    with open(output_path, "w") as f:
        f.write("# Task Distribution Analysis\n\n")

        for subset, dist in distributions.items():
            f.write(f"## {subset.capitalize()} Subset\n\n")
            f.write(f"Total tasks: {dist['total_tasks']}\n\n")

            # Difficulty distribution
            f.write("### Difficulty Distribution\n\n")
            for diff, pct in sorted(dist['difficulty_distribution'].items()):
                count = dist['difficulty_counts'][diff]
                f.write(f"- **{diff.capitalize()}**: {count} tasks ({pct*100:.1f}%)\n")
            f.write("\n")

            # Category distribution
            f.write("### Category Distribution\n\n")
            for cat, pct in sorted(dist['category_distribution'].items(), key=lambda x: -x[1]):
                count = dist['category_counts'][cat]
                f.write(f"- **{cat.replace('_', ' ').title()}**: {count} tasks ({pct*100:.1f}%)\n")
            f.write("\n")

    logger.info(f"Distribution report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Task Analyzer Module")
    logger.info("Classifies tasks by difficulty and category")
