"""Evaluation harness for SWE-bench tasks."""
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from loguru import logger

from src.models import TaskResult, EvaluationResults, Trajectory
from src.swebench_agent import SWEBenchAgent
from src.swebench_loader import SWEBenchDataset, SWEBenchTask
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.memory import ReasoningBank


class SWEBenchEvaluator:
    """Evaluator for SWE-bench tasks."""

    def __init__(
        self,
        agent: SWEBenchAgent,
        judge: Optional[TrajectoryJudge] = None,
        extractor: Optional[StrategyExtractor] = None,
        memory_bank: Optional[ReasoningBank] = None,
        output_dir: str = "results",
        log_dir: str = "logs",
        checkpoint_interval: int = 10,
        mode: str = "no_memory"
    ):
        """
        Initialize evaluator.

        Args:
            agent: SWE-bench agent
            judge: Optional trajectory judge for success/failure detection
            extractor: Optional strategy extractor for memory
            memory_bank: Optional ReasoningBank
            output_dir: Directory for results
            log_dir: Directory for logs
            checkpoint_interval: Save checkpoint every N tasks
            mode: Evaluation mode (no_memory, synapse, reasoningbank)
        """
        self.agent = agent
        self.judge = judge
        self.extractor = extractor
        self.memory_bank = memory_bank
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_interval = checkpoint_interval
        self.mode = mode

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_dataset(
        self,
        dataset: SWEBenchDataset,
        seed: int = 42,
        max_tasks: Optional[int] = None,
        save_trajectories: bool = True
    ) -> EvaluationResults:
        """
        Evaluate agent on SWE-bench dataset.

        Args:
            dataset: SWE-bench dataset
            seed: Random seed
            max_tasks: Optional max number of tasks to evaluate
            save_trajectories: Whether to save trajectory files

        Returns:
            Evaluation results
        """
        logger.info(f"Starting SWE-bench evaluation with {len(dataset)} tasks")

        # Get tasks (already filtered by loader)
        tasks = dataset.tasks

        # Results storage
        results: List[TaskResult] = []

        # Evaluate each task
        for idx, task in enumerate(tqdm(tasks, desc="Evaluating SWE-bench")):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {idx+1}/{len(tasks)}: {task.instance_id}")
            logger.info(f"{'='*60}")

            # Create per-task log file for detailed logging
            task_log_file = self.log_dir / f"task_{task.instance_id}.log"
            task_logger_id = logger.add(
                task_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                level="DEBUG",  # Capture more detailed logs for each task
                filter=lambda record: record["extra"].get("task_id") == task.instance_id
            )

            try:
                # Run task with task-specific logging context
                with logger.contextualize(task_id=task.instance_id):
                    logger.info(f"Starting task: {task.instance_id}")
                    logger.info(f"Repository: {task.repo}")
                    logger.info(f"Base commit: {task.base_commit}")

                    result = self._evaluate_task(
                        task=task,
                        seed=seed,
                        save_trajectory=save_trajectories
                    )
                    results.append(result)

                    # Log result with context
                    status = "✓ SUCCESS" if result.success else "✗ FAILURE"
                    logger.info(f"{status} | Steps: {result.steps} | Time: {result.walltime:.1f}s")
                    logger.info(f"Task completed: {task.instance_id}")

                # Save results continuously after each task
                self._save_incremental_results(results, idx + 1)

                # Also save memory bank checkpoint if using ReasoningBank
                if self.memory_bank and (idx + 1) % 10 == 0:
                    self.memory_bank.save_checkpoint()
                    logger.info(f"Memory bank checkpoint saved at task {idx + 1}")

            except Exception as e:
                with logger.contextualize(task_id=task.instance_id):
                    logger.error(f"Error evaluating task {task.instance_id}: {e}")
                    logger.exception("Full traceback:")

                # Create error result
                error_result = TaskResult(
                    task_id=task.instance_id,
                    subset="swebench",
                    success=False,
                    steps=0,
                    tokens_input=0,
                    tokens_output=0,
                    walltime=0.0,
                    seed=seed,
                    error=str(e)
                )
                results.append(error_result)

                # Save results even after errors
                self._save_incremental_results(results, idx + 1)

            finally:
                # Remove task-specific logger handler
                logger.remove(task_logger_id)

        # Compute aggregated results
        eval_results = self._aggregate_results(results, subset="test")

        # Save final results
        self._save_results(results, eval_results)

        return eval_results

    def _evaluate_task(
        self,
        task: SWEBenchTask,
        seed: int,
        save_trajectory: bool
    ) -> TaskResult:
        """
        Evaluate a single SWE-bench task.

        Args:
            task: SWE-bench task instance
            seed: Random seed
            save_trajectory: Whether to save trajectory

        Returns:
            Task result
        """
        # Setup repository
        repo_path = self._setup_repository(task)

        try:
            # Get task description
            task_description = task.get_task_description()

            # Run agent
            trajectory = self.agent.run_task(
                task_description=task_description,
                repo_path=repo_path,
                base_commit=task.base_commit,
                seed=seed
            )

            # Evaluate the result
            success = self._evaluate_patch(
                task=task,
                patch=trajectory.final_answer,
                repo_path=repo_path
            )

            # Update trajectory
            trajectory.success = success
            trajectory.task_description = task_description
            trajectory.task_id = task.instance_id
            trajectory.reference_answer = task.gold_patch

            # Extract memories if using ReasoningBank
            if self.memory_bank and self.judge and self.extractor:
                try:
                    # Use judge to get success/failure signal
                    judgment = self.judge.judge(
                        trajectory=trajectory,
                        task_description=task_description
                    )

                    # Extract strategies
                    memories = self.extractor.extract(
                        trajectory=trajectory,
                        task_description=task_description,
                        success=judgment
                    )

                    # Add to memory bank
                    for memory in memories:
                        self.memory_bank.add_memory(memory)

                    logger.info(f"Extracted {len(memories)} memories from trajectory")

                except Exception as e:
                    logger.warning(f"Failed to extract memories: {e}")

            # Save trajectory if requested
            if save_trajectory:
                traj_path = self.log_dir / f"trajectory_{task.instance_id}.json"
                with open(traj_path, "w") as f:
                    json.dump(trajectory.model_dump(), f, indent=2)

            # Create result
            result = TaskResult(
                task_id=task.instance_id,
                subset="swebench",
                success=success,
                steps=trajectory.steps,
                tokens_input=trajectory.tokens.get("input", 0),
                tokens_output=trajectory.tokens.get("output", 0),
                walltime=trajectory.walltime,
                seed=seed,
                trajectory_path=str(traj_path) if save_trajectory else None,
                agent_answer=trajectory.final_answer,
                reference_answer=task.gold_patch
            )

            return result

        finally:
            # Cleanup repository
            self._cleanup_repository(repo_path)

    def _setup_repository(self, task: SWEBenchTask) -> str:
        """
        Setup repository for task evaluation.

        Args:
            task: SWE-bench task

        Returns:
            Path to repository
        """
        # Create temporary directory for repository
        temp_dir = tempfile.mkdtemp(prefix="swebench_")
        logger.info(f"Setting up repository in {temp_dir}")

        try:
            # Clone repository
            repo_url = self._get_repo_url(task.repo)
            subprocess.run(
                ["git", "clone", repo_url, temp_dir],
                check=True,
                capture_output=True
            )

            # Checkout base commit
            subprocess.run(
                ["git", "checkout", task.base_commit],
                cwd=temp_dir,
                check=True,
                capture_output=True
            )

            logger.info(f"Repository ready: {task.repo} @ {task.base_commit[:8]}")
            return temp_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup repository: {e}")
            raise

    def _cleanup_repository(self, repo_path: str):
        """Cleanup repository directory."""
        try:
            shutil.rmtree(repo_path)
            logger.debug(f"Cleaned up repository: {repo_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup repository {repo_path}: {e}")

    def _get_repo_url(self, repo: str) -> str:
        """
        Get GitHub URL for repository.

        Args:
            repo: Repository name (e.g., "django/django")

        Returns:
            GitHub clone URL
        """
        return f"https://github.com/{repo}.git"

    def _normalize_patch(self, patch: str) -> str:
        """Normalize patch for comparison by removing formatting differences."""
        if not patch:
            return ""

        lines = []
        for line in patch.split('\n'):
            # Skip index lines
            if line.startswith('index '):
                continue
            # Skip empty lines at start/end
            stripped = line.rstrip()
            if stripped or lines:  # Keep internal empty lines
                lines.append(stripped)

        # Remove trailing empty lines
        while lines and not lines[-1]:
            lines.pop()

        return '\n'.join(lines)

    def _evaluate_patch(
        self,
        task: SWEBenchTask,
        patch: Optional[str],
        repo_path: str
    ) -> bool:
        """
        Evaluate if patch fixes the issue.

        Args:
            task: SWE-bench task
            patch: Generated patch
            repo_path: Repository path

        Returns:
            True if patch passes tests
        """
        if not patch:
            logger.warning("No patch generated")
            return False

        # First, check if patch matches reference semantically
        if task.gold_patch:
            normalized_agent = self._normalize_patch(patch)
            normalized_gold = self._normalize_patch(task.gold_patch)

            if normalized_agent == normalized_gold:
                logger.info("✓ Patch matches reference answer exactly!")
                return True

            # Check if the key change lines match (more lenient)
            agent_changes = [l for l in normalized_agent.split('\n') if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
            gold_changes = [l for l in normalized_gold.split('\n') if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]

            if agent_changes == gold_changes:
                logger.info("✓ Patch has correct changes (matches reference)")
                return True

        # Clean patch: remove index lines that cause git apply issues
        patch_lines = patch.split('\n')
        cleaned_lines = []
        for line in patch_lines:
            # Skip index lines which can cause "corrupt patch" errors
            if line.startswith('index '):
                continue
            cleaned_lines.append(line)
        patch = '\n'.join(cleaned_lines)

        try:
            # Apply patch
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch)
                patch_file = f.name

            try:
                subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )
                logger.info("Patch applied successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to apply patch: {e.stderr.decode()}")
                return False
            finally:
                Path(patch_file).unlink()

            # Run tests
            # Note: This is simplified - real SWE-bench evaluation is more complex
            # In production, you'd use the official SWE-bench harness
            logger.info("Running tests...")

            # Try common test commands
            test_commands = [
                "python -m pytest",
                "python -m unittest discover",
                "python setup.py test"
            ]

            for cmd in test_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=repo_path,
                        capture_output=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        logger.info("Tests passed")
                        return True
                    else:
                        logger.warning(f"Tests failed with command: {cmd}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Tests timed out with command: {cmd}")
                except Exception as e:
                    logger.warning(f"Error running tests with {cmd}: {e}")

            return False

        except Exception as e:
            logger.error(f"Error evaluating patch: {e}")
            return False

    def _aggregate_results(
        self,
        results: List[TaskResult],
        subset: str = "test"
    ) -> EvaluationResults:
        """Aggregate results across all tasks."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        total_steps = sum(r.steps for r in results if r.success)
        total_tokens = sum(r.tokens_input + r.tokens_output for r in results)
        total_walltime = sum(r.walltime for r in results)

        avg_steps = total_steps / successful_tasks if successful_tasks > 0 else 0.0
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        return EvaluationResults(
            mode=self.mode,
            subset=subset,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            success_rate=success_rate,
            avg_steps=avg_steps,
            total_tokens=total_tokens,
            total_walltime=total_walltime,
            task_results=results
        )

    def _save_incremental_results(self, results: List[TaskResult], task_num: int):
        """
        Save results incrementally after each task.

        This ensures progress is saved continuously and not lost if the run is interrupted.
        """
        # Save all results so far to results.json
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)

        # Compute and save current summary statistics
        if results:
            eval_results = self._aggregate_results(results, subset="test")
            summary_path = self.output_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(eval_results.model_dump(), f, indent=2)

        logger.debug(f"Saved progress: {task_num} tasks completed")

    def _save_results(
        self,
        results: List[TaskResult],
        eval_results: EvaluationResults
    ):
        """Save final results."""
        # Save individual results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)

        # Save aggregated results
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(eval_results.model_dump(), f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
