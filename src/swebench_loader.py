"""SWE-bench dataset loader for ReasoningBank.

Loads SWE-Bench-Verified dataset for repository-level issue resolution.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datasets import load_dataset


class SWEBenchTask:
    """Single SWE-bench task instance."""

    def __init__(
        self,
        instance_id: str,
        repo: str,
        base_commit: str,
        problem_statement: str,
        hints_text: str,
        test_patch: str,
        patch: str,
        version: str,
        **kwargs
    ):
        self.instance_id = instance_id
        self.repo = repo
        self.base_commit = base_commit
        self.problem_statement = problem_statement
        self.hints_text = hints_text
        self.test_patch = test_patch
        self.gold_patch = patch
        self.version = version
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "test_patch": self.test_patch,
            "gold_patch": self.gold_patch,
            "version": self.version,
            **self.metadata
        }

    def get_task_description(self) -> str:
        """Get formatted task description for agent."""
        desc = f"Repository: {self.repo}\n"
        desc += f"Issue ID: {self.instance_id}\n\n"
        desc += f"Problem Statement:\n{self.problem_statement}\n"

        if self.hints_text:
            desc += f"\nHints:\n{self.hints_text}\n"

        return desc


class SWEBenchDataset:
    """SWE-Bench-Verified dataset loader."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        split: str = "test"
    ):
        """
        Initialize SWE-bench dataset.

        Args:
            cache_dir: Optional cache directory for dataset
            split: Dataset split to load (default: "test")
        """
        self.cache_dir = cache_dir
        self.split = split
        self.tasks: List[SWEBenchTask] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load SWE-Bench-Verified from HuggingFace."""
        logger.info("Loading SWE-Bench-Verified dataset...")

        try:
            # Load from HuggingFace datasets
            dataset = load_dataset(
                "princeton-nlp/SWE-bench_Verified",
                split=self.split,
                cache_dir=self.cache_dir
            )

            logger.info(f"Loaded {len(dataset)} instances from SWE-Bench-Verified")

            # Convert to SWEBenchTask objects
            for item in dataset:
                task = SWEBenchTask(**item)
                self.tasks.append(task)

            logger.info(f"Successfully loaded {len(self.tasks)} SWE-bench tasks")

            # Log repository distribution
            repos = {}
            for task in self.tasks:
                repos[task.repo] = repos.get(task.repo, 0) + 1

            logger.info("Repository distribution:")
            for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
                logger.info(f"  {repo}: {count} instances")

        except Exception as e:
            logger.error(f"Failed to load SWE-Bench-Verified: {e}")
            logger.error("Please install required packages: pip install datasets")
            raise

    def get_task(self, instance_id: str) -> Optional[SWEBenchTask]:
        """Get task by instance ID."""
        for task in self.tasks:
            if task.instance_id == instance_id:
                return task
        return None

    def get_tasks_by_repo(self, repo: str) -> List[SWEBenchTask]:
        """Get all tasks for a specific repository."""
        return [t for t in self.tasks if t.repo == repo]

    def filter_tasks(
        self,
        repos: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        start_index: int = 0
    ) -> List[SWEBenchTask]:
        """
        Filter tasks by repository and limit.

        Args:
            repos: List of repositories to include (None = all)
            max_tasks: Maximum number of tasks to return
            start_index: Starting index for task selection (for parallel processing)

        Returns:
            Filtered list of tasks
        """
        tasks = self.tasks

        # Filter by repository
        if repos:
            tasks = [t for t in tasks if t.repo in repos]
            logger.info(f"Filtered to {len(tasks)} tasks from {len(repos)} repositories")

        # Apply start_index and max_tasks for slicing
        if start_index > 0:
            tasks = tasks[start_index:]
            logger.info(f"Starting from index {start_index}")

        # Limit number of tasks
        if max_tasks and max_tasks < len(tasks):
            tasks = tasks[:max_tasks]
            logger.info(f"Limited to {max_tasks} tasks")

        return tasks

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> SWEBenchTask:
        """Get task by index."""
        return self.tasks[idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        repos = {}
        for task in self.tasks:
            repos[task.repo] = repos.get(task.repo, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "num_repos": len(repos),
            "repos": repos
        }


def load_swebench_dataset(
    cache_dir: Optional[str] = None,
    split: str = "test",
    repos: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    start_index: int = 0
) -> SWEBenchDataset:
    """
    Convenience function to load SWE-bench dataset.

    Args:
        cache_dir: Optional cache directory
        split: Dataset split (default: "test")
        repos: Optional list of repositories to filter
        max_tasks: Optional maximum number of tasks
        start_index: Starting index for task selection (for parallel processing)

    Returns:
        SWEBenchDataset instance
    """
    dataset = SWEBenchDataset(cache_dir=cache_dir, split=split)

    if repos or max_tasks or start_index > 0:
        # Apply filtering and update dataset tasks
        filtered = dataset.filter_tasks(repos=repos, max_tasks=max_tasks, start_index=start_index)
        dataset.tasks = filtered
        logger.info(f"Dataset ready with {len(filtered)} tasks (start_index={start_index})")

    return dataset
