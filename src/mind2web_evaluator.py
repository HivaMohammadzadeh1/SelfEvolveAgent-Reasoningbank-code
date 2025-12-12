"""Evaluation harness for Mind2Web tasks.

Implements Mind2Web evaluation metrics:
- Element Accuracy (EA): Accuracy of element selection
- Action F1 (AF1): F1 score for action prediction
- Step Success Rate (SSR): Step-level success rate (element + action correct)
- Success Rate (SR): Task-level success rate
"""
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
from loguru import logger

from src.models import TaskResult, EvaluationResults, Trajectory
from src.mind2web_agent import Mind2WebAgent
from src.mind2web_loader import Mind2WebDataset, Mind2WebTask
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.memory import ReasoningBank


class Mind2WebEvaluator:
    """Evaluator for Mind2Web tasks with benchmark-specific metrics."""

    def __init__(
        self,
        agent: Mind2WebAgent,
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
            agent: Mind2Web agent
            judge: Optional trajectory judge for success/failure detection
            extractor: Optional strategy extractor for memory
            memory_bank: Optional ReasoningBank
            output_dir: Directory for results
            log_dir: Directory for logs
            checkpoint_interval: Save checkpoint every N tasks
            mode: Evaluation mode (no_memory, synapse, awm, reasoningbank)
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
        dataset: Mind2WebDataset,
        seed: int = 42,
        max_tasks: Optional[int] = None,
        save_trajectories: bool = True
    ) -> EvaluationResults:
        """
        Evaluate agent on Mind2Web dataset.

        Args:
            dataset: Mind2Web dataset
            seed: Random seed
            max_tasks: Optional max number of tasks to evaluate
            save_trajectories: Whether to save trajectory files

        Returns:
            Evaluation results with Mind2Web metrics
        """
        logger.info(f"Starting Mind2Web evaluation with {len(dataset)} tasks")
        logger.info(f"Split: {dataset.SPLIT_NAMES.get(dataset.split, dataset.split)}")

        # Get tasks
        tasks = dataset.tasks

        # Results storage
        results: List[TaskResult] = []
        all_metrics = []  # Store per-task metrics for aggregation

        # Evaluate each task
        for idx, task in enumerate(tqdm(tasks, desc="Evaluating Mind2Web")):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {idx+1}/{len(tasks)}: {task.annotation_id}")
            logger.info(f"Website: {task.website}, Domain: {task.domain}")
            logger.info(f"{'='*60}")

            try:
                # Run task
                result, metrics = self._evaluate_task(
                    task=task,
                    seed=seed,
                    save_trajectory=save_trajectories
                )
                results.append(result)
                all_metrics.append(metrics)

                # Log result
                status = "✓ SUCCESS" if result.success else "✗ FAILURE"
                logger.info(f"{status} | Steps: {result.steps} | Time: {result.walltime:.1f}s")
                logger.info(f"Metrics: EA={metrics['element_accuracy']:.1%}, "
                          f"AF1={metrics['action_f1']:.1%}, "
                          f"SSR={metrics['step_success_rate']:.1%}")

                # Save checkpoint
                if (idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(results, idx + 1)
                    logger.info(f"Checkpoint saved at task {idx + 1}")

            except Exception as e:
                logger.error(f"Error evaluating task {task.annotation_id}: {e}")
                # Create error result
                error_result = TaskResult(
                    task_id=task.annotation_id,
                    subset=dataset.split,
                    success=False,
                    steps=0,
                    tokens_input=0,
                    tokens_output=0,
                    walltime=0.0,
                    seed=seed,
                    error=str(e)
                )
                results.append(error_result)
                all_metrics.append({
                    "element_accuracy": 0.0,
                    "action_f1": 0.0,
                    "step_success_rate": 0.0
                })

        # Compute aggregated results with Mind2Web metrics
        eval_results = self._aggregate_results(results, all_metrics, dataset.split)

        # Save final results
        self._save_results(results, eval_results, all_metrics)

        return eval_results

    def _evaluate_task(
        self,
        task: Mind2WebTask,
        seed: int,
        save_trajectory: bool
    ) -> Tuple[TaskResult, Dict[str, float]]:
        """
        Evaluate a single Mind2Web task.

        Args:
            task: Mind2Web task instance
            seed: Random seed
            save_trajectory: Whether to save trajectory

        Returns:
            Tuple of (TaskResult, metrics_dict)
        """
        # Get task data
        task_data = task.to_dict()

        # Run agent
        trajectory = self.agent.run_task(
            task_data=task_data,
            seed=seed
        )

        # Compute Mind2Web metrics
        metrics = self._compute_mind2web_metrics(
            predicted_actions=trajectory.predicted_actions,
            ground_truth_actions=task.actions,
            trajectory=trajectory
        )

        # Task success based on completion and metrics
        # In Mind2Web, success is typically based on step success rate
        success = metrics["step_success_rate"] > 0.5  # Threshold can be adjusted

        # Update trajectory
        trajectory.success = success
        trajectory.task_description = task.get_task_description()
        trajectory.task_id = task.annotation_id
        # Use action_reprs as reference answer (human-readable action sequence)
        trajectory.reference_answer = "\n".join(task.action_reprs) if task.action_reprs else ""

        # Extract memories if using ReasoningBank
        if self.memory_bank and self.judge and self.extractor:
            try:
                # Use judge to get success/failure signal
                judgment = self.judge.judge(
                    trajectory=trajectory,
                    task_description=task.get_task_description()
                )

                # Extract strategies
                memories = self.extractor.extract(
                    trajectory=trajectory,
                    task_description=task.get_task_description(),
                    success=judgment
                )

                # Add to memory bank
                for memory in memories:
                    self.memory_bank.add_memory(memory)

                logger.info(f"Extracted {len(memories)} memories from trajectory")

            except Exception as e:
                logger.warning(f"Failed to extract memories: {e}")

        # Save trajectory if requested
        traj_path = None
        if save_trajectory:
            # Save trajectories in mode-specific subdirectory
            traj_dir = self.log_dir / f"mind2web_trajectories_{self.mode}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_path = traj_dir / f"trajectory_{task.annotation_id}.json"
            with open(traj_path, "w") as f:
                traj_dict = trajectory.model_dump()
                traj_dict["mind2web_metrics"] = metrics
                json.dump(traj_dict, f, indent=2)

        # Create result
        result = TaskResult(
            task_id=task.annotation_id,
            subset=task_data.get("subdomain", "unknown"),
            success=success,
            steps=trajectory.steps,
            tokens_input=trajectory.tokens.get("input", 0),
            tokens_output=trajectory.tokens.get("output", 0),
            walltime=trajectory.walltime,
            seed=seed,
            trajectory_path=str(traj_path) if save_trajectory else None,
            agent_answer=trajectory.final_answer,
            reference_answer="\n".join(task.action_reprs) if task.action_reprs else ""
        )

        return result, metrics

    def _compute_mind2web_metrics(
        self,
        predicted_actions: List[Dict[str, Any]],
        ground_truth_actions: List[Dict[str, Any]],
        trajectory: Trajectory
    ) -> Dict[str, float]:
        """
        Compute Mind2Web evaluation metrics.

        Metrics:
        - Element Accuracy (EA): Fraction of steps with correct element
        - Action F1 (AF1): F1 score for action prediction
        - Step Success Rate (SSR): Fraction of steps with correct element AND action
        - Success Rate (SR): Whether the full task is completed successfully

        Args:
            predicted_actions: Agent's predicted actions
            ground_truth_actions: Ground truth action sequence
            trajectory: Complete trajectory

        Returns:
            Dictionary with metrics
        """
        if not ground_truth_actions or len(ground_truth_actions) == 0:
            # No ground truth available, use heuristic evaluation
            return {
                "element_accuracy": 1.0 if trajectory.success else 0.0,
                "action_f1": 1.0 if trajectory.success else 0.0,
                "step_success_rate": 1.0 if trajectory.success else 0.0,
                "success_rate": 1.0 if trajectory.success else 0.0
            }

        # Align predicted and ground truth actions
        num_gt_steps = len(ground_truth_actions)
        num_pred_steps = min(len(predicted_actions), num_gt_steps)

        # Count correct elements and actions
        correct_elements = 0
        correct_actions = 0
        correct_steps = 0

        # For F1 computation
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(num_pred_steps):
            pred = predicted_actions[i]
            gt = ground_truth_actions[i]

            pred_action = pred.get("action_type", "").lower()
            gt_action = self._extract_ground_truth_action(gt)

            # Element accuracy: Check if element is correct
            # In Mind2Web, this is typically based on element attributes
            element_match = self._check_element_match(pred, gt)
            if element_match:
                correct_elements += 1

            # Action accuracy: Check if action type is correct
            action_match = pred_action == gt_action
            if action_match:
                correct_actions += 1
                true_positives += 1
            else:
                if pred_action:
                    false_positives += 1
                false_negatives += 1

            # Step success: Both element and action correct
            if element_match and action_match:
                correct_steps += 1

        # Compute metrics
        element_accuracy = correct_elements / num_gt_steps if num_gt_steps > 0 else 0.0

        # Action F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        action_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        step_success_rate = correct_steps / num_gt_steps if num_gt_steps > 0 else 0.0

        # Success rate: Full task completion (heuristic)
        # Consider successful if SSR > 0.8 or agent explicitly finished successfully
        success_rate = 1.0 if (step_success_rate > 0.8 or trajectory.success) else 0.0

        return {
            "element_accuracy": element_accuracy,
            "action_f1": action_f1,
            "step_success_rate": step_success_rate,
            "success_rate": success_rate
        }

    def _extract_ground_truth_action(self, gt_action: Dict[str, Any]) -> str:
        """Extract action type from ground truth action dictionary."""
        # Mind2Web ground truth format: {"action_type": "click", ...}
        # or {"operation": {"op_type": "click"}}
        if "action_type" in gt_action:
            return gt_action["action_type"].lower()
        elif "operation" in gt_action and isinstance(gt_action["operation"], dict):
            return gt_action["operation"].get("op_type", "").lower()
        elif "op_type" in gt_action:
            return gt_action["op_type"].lower()
        else:
            # Try to infer from action structure
            return "unknown"

    def _check_element_match(self, pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
        """
        Check if predicted element matches ground truth element.

        This is a simplified check. Full Mind2Web evaluation uses DOM element matching.
        """
        # Extract element identifiers
        pred_elem = pred.get("args", {}).get("element_id", "")

        # Ground truth element can be in various formats
        gt_elem = None
        if "pos_candidates" in gt and gt["pos_candidates"]:
            # Mind2Web format: pos_candidates contains target element info
            gt_elem = str(gt["pos_candidates"][0].get("backend_node_id", ""))
        elif "element_id" in gt:
            gt_elem = str(gt["element_id"])
        elif "target" in gt:
            gt_elem = str(gt["target"].get("backend_node_id", ""))

        # Simple string match (in practice, this should use DOM matching)
        if gt_elem and pred_elem:
            return str(pred_elem) == str(gt_elem)

        # If we can't determine, be lenient
        return False

    def _aggregate_results(
        self,
        results: List[TaskResult],
        all_metrics: List[Dict[str, float]],
        subset: str
    ) -> EvaluationResults:
        """Aggregate results across all tasks with Mind2Web metrics."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        total_steps = sum(r.steps for r in results)
        total_tokens = sum(r.tokens_input + r.tokens_output for r in results)
        total_walltime = sum(r.walltime for r in results)

        avg_steps = total_steps / total_tasks if total_tasks > 0 else 0.0
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        # Aggregate Mind2Web metrics
        avg_element_accuracy = sum(m["element_accuracy"] for m in all_metrics) / total_tasks if total_tasks > 0 else 0.0
        avg_action_f1 = sum(m["action_f1"] for m in all_metrics) / total_tasks if total_tasks > 0 else 0.0
        avg_step_success_rate = sum(m["step_success_rate"] for m in all_metrics) / total_tasks if total_tasks > 0 else 0.0

        # Log Mind2Web metrics
        logger.info(f"\nMind2Web Metrics ({subset}):")
        logger.info(f"  Element Accuracy (EA):    {avg_element_accuracy:.1%}")
        logger.info(f"  Action F1 (AF1):          {avg_action_f1:.1%}")
        logger.info(f"  Step Success Rate (SSR):  {avg_step_success_rate:.1%}")
        logger.info(f"  Success Rate (SR):        {success_rate:.1%}")

        # Create evaluation results with Mind2Web metrics
        eval_results = EvaluationResults(
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

        # Add Mind2Web-specific metrics
        eval_results.mind2web_metrics = {
            "element_accuracy": avg_element_accuracy,
            "action_f1": avg_action_f1,
            "step_success_rate": avg_step_success_rate,
            "task_success_rate": success_rate
        }

        return eval_results

    def _save_checkpoint(self, results: List[TaskResult], task_num: int):
        """Save checkpoint of results."""
        checkpoint_path = self.output_dir / f"checkpoint_{task_num}.json"
        with open(checkpoint_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)

    def _save_results(
        self,
        results: List[TaskResult],
        eval_results: EvaluationResults,
        all_metrics: List[Dict[str, float]]
    ):
        """Save final results."""
        # Save individual results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            results_data = []
            for r, m in zip(results, all_metrics):
                r_dict = r.model_dump()
                r_dict["mind2web_metrics"] = m
                results_data.append(r_dict)
            json.dump(results_data, f, indent=2)

        # Save aggregated results
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(eval_results.model_dump(), f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
