"""
Task analyzer for SWE-bench tasks.

Provides classification of tasks by:
- Difficulty (easy, medium, hard)
- Issue type (bug_fix, feature_request, performance, documentation, other)
- Repository characteristics
"""

from typing import Dict, Any, List
import re


class SWEBenchTaskAnalyzer:
    """Analyzer for classifying SWE-bench tasks."""

    def __init__(self):
        """Initialize task analyzer."""
        self.difficulty_keywords = {
            "easy": [
                "typo", "docstring", "comment", "spelling", "formatting",
                "print", "log", "simple", "straightforward", "obvious"
            ],
            "hard": [
                "race condition", "deadlock", "memory leak", "concurrency",
                "optimization", "refactor", "architecture", "complex",
                "subtle", "edge case", "regression", "intermittent"
            ]
        }

        self.issue_type_keywords = {
            "bug_fix": [
                "bug", "fix", "error", "exception", "crash", "fail",
                "incorrect", "wrong", "broken", "issue", "problem"
            ],
            "feature_request": [
                "feature", "add", "implement", "support", "enhancement",
                "new", "allow", "enable", "provide"
            ],
            "performance": [
                "performance", "slow", "optimization", "optimize", "faster",
                "speed", "efficient", "memory", "cpu", "bottleneck"
            ],
            "documentation": [
                "documentation", "doc", "docstring", "readme", "comment",
                "typo", "spelling", "example", "tutorial"
            ]
        }

    def classify_difficulty(self, task_result: Dict[str, Any]) -> str:
        """
        Classify task difficulty based on various factors.

        Args:
            task_result: Task result dictionary

        Returns:
            Difficulty level: "easy", "medium", or "hard"
        """
        # Factors for difficulty:
        # 1. Number of steps taken
        # 2. Whether task was successful
        # 3. Keywords in task description
        # 4. Number of files modified (if available)

        steps = task_result.get("steps", 0)
        success = task_result.get("success", False)

        # Get task description from trajectory if available
        task_description = ""
        trajectory_path = task_result.get("trajectory_path")
        if trajectory_path:
            try:
                import json
                from pathlib import Path
                with open(trajectory_path, "r") as f:
                    trajectory = json.load(f)
                    task_description = trajectory.get("task_description", "")
            except:
                pass

        # Check for easy keywords
        task_text = (task_description or task_result.get("task_id", "")).lower()
        easy_score = sum(1 for keyword in self.difficulty_keywords["easy"] if keyword in task_text)
        hard_score = sum(1 for keyword in self.difficulty_keywords["hard"] if keyword in task_text)

        # Heuristic classification
        if easy_score > 0 and steps < 15:
            return "easy"
        elif hard_score > 0 or steps > 25:
            return "hard"
        elif steps > 20 or (steps > 15 and not success):
            return "hard"
        elif steps < 10:
            return "easy"
        else:
            return "medium"

    def classify_issue_type(self, task_result: Dict[str, Any]) -> str:
        """
        Classify issue type based on task description.

        Args:
            task_result: Task result dictionary

        Returns:
            Issue type: "bug_fix", "feature_request", "performance", "documentation", or "other"
        """
        # Get task description from trajectory if available
        task_description = ""
        trajectory_path = task_result.get("trajectory_path")
        if trajectory_path:
            try:
                import json
                from pathlib import Path
                with open(trajectory_path, "r") as f:
                    trajectory = json.load(f)
                    task_description = trajectory.get("task_description", "")
            except:
                pass

        task_text = (task_description or task_result.get("task_id", "")).lower()

        # Count keyword matches for each type
        type_scores = {}
        for issue_type, keywords in self.issue_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_text)
            type_scores[issue_type] = score

        # Return type with highest score
        if sum(type_scores.values()) == 0:
            return "other"

        return max(type_scores.items(), key=lambda x: x[1])[0]

    def extract_repo_info(self, task_id: str) -> Dict[str, str]:
        """
        Extract repository information from task ID.

        Args:
            task_id: Task ID (format: owner__repo-issue_number)

        Returns:
            Dictionary with owner, repo, and issue_number
        """
        # Task IDs are typically in format: owner__repo-issue_number
        # or owner-repo-issue_number
        task_id = task_id.replace("__", "/")

        match = re.match(r"([^/]+)/([^-]+)-(.+)", task_id)
        if match:
            return {
                "owner": match.group(1),
                "repo": match.group(2),
                "issue_number": match.group(3),
                "full_repo": f"{match.group(1)}/{match.group(2)}"
            }

        return {
            "owner": "unknown",
            "repo": "unknown",
            "issue_number": "unknown",
            "full_repo": "unknown"
        }

    def analyze_repository_complexity(self, repo_name: str) -> str:
        """
        Estimate repository complexity based on known repositories.

        Args:
            repo_name: Repository name (e.g., "django/django")

        Returns:
            Complexity level: "high", "medium", or "low"
        """
        # Known complex repositories
        high_complexity = [
            "django/django",
            "scikit-learn/scikit-learn",
            "sympy/sympy",
            "matplotlib/matplotlib",
            "sphinx-doc/sphinx"
        ]

        if repo_name in high_complexity:
            return "high"

        # Default to medium
        return "medium"


# Convenience functions
def classify_task_difficulty(task_result: Dict[str, Any]) -> str:
    """
    Classify task difficulty.

    Args:
        task_result: Task result dictionary

    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    analyzer = SWEBenchTaskAnalyzer()
    return analyzer.classify_difficulty(task_result)


def classify_issue_type(task_result: Dict[str, Any]) -> str:
    """
    Classify issue type.

    Args:
        task_result: Task result dictionary

    Returns:
        Issue type: "bug_fix", "feature_request", "performance", "documentation", or "other"
    """
    analyzer = SWEBenchTaskAnalyzer()
    return analyzer.classify_issue_type(task_result)


def extract_repo_from_task_id(task_id: str) -> str:
    """
    Extract repository name from task ID.

    Args:
        task_id: Task ID

    Returns:
        Repository name (e.g., "django/django")
    """
    analyzer = SWEBenchTaskAnalyzer()
    repo_info = analyzer.extract_repo_info(task_id)
    return repo_info["full_repo"]
