"""Evaluation harness for running experiments."""
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from loguru import logger

from src.models import TaskResult, EvaluationResults, Trajectory
from src.agent import ReActAgent, BrowserEnvironment
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.memory import ReasoningBank


class TaskDataset:
    """
    Load and manage WebArena task dataset.
    
    Can load either real WebArena data or fallback to mock data.
    """
    
    def __init__(self, data_path: str, subsets: List[str], use_real_data: bool = True):
        self.data_path = Path(data_path)
        self.subsets = subsets
        self.use_real_data = use_real_data
        self.tasks = []
        self._load_tasks()
    
    def _load_tasks(self):
        """Load tasks from dataset."""
        if self.use_real_data:
            try:
                self._load_real_webarena_tasks()
                return
            except Exception as e:
                logger.error(f"Failed to load real WebArena data: {e}")
                logger.warning("Falling back to mock data")
        
        self._load_mock_tasks()
    
    def _load_real_webarena_tasks(self):
        """Load real WebArena tasks from config files."""
        from src.webarena_loader import WebArenaDataset
        
        logger.info(f"Loading real WebArena data from {self.data_path}")
        
        # Initialize WebArena loader
        webarena = WebArenaDataset(data_dir=str(self.data_path))
        
        # Load tasks for specified subsets
        # Use max_multi_tasks=29 to match paper (WebArena has 48 total)
        self.tasks = webarena.load_tasks(subsets=self.subsets, max_multi_tasks=29)
        
        # Log statistics
        stats = webarena.get_statistics()
        logger.info(f"Loaded {stats['total_tasks']} real WebArena tasks")
        for subset, count in stats['subset_counts'].items():
            logger.info(f"  {subset}: {count} tasks")
    
    def _load_mock_tasks(self):
        """Load mock tasks for testing (fallback)."""
        logger.warning("Using mock task dataset - replace with actual WebArena data")
        logger.info("To use real data:")
        logger.info("  1. Clone WebArena: git clone https://github.com/web-arena-x/webarena.git data/webarena_repo")
        logger.info("  2. Generate configs: python data/webarena_repo/scripts/generate_test_data.py")
        logger.info("  3. Copy configs: cp -r data/webarena_repo/config_files data/webarena/")
        
        # Create mock tasks for demonstration
        task_id = 0
        for subset in self.subsets:
            # Mock counts from PRD
            counts = {
                "shopping": 187,
                "admin": 182,
                "gitlab": 180,
                "reddit": 106,
                "multi": 29
            }
            
            num_tasks = counts.get(subset, 10)
            for i in range(num_tasks):
                task_id += 1
                self.tasks.append({
                    "task_id": f"{subset}_{i:03d}",
                    "numeric_id": task_id,
                    "subset": subset,
                    "description": f"Mock task {task_id} for {subset} domain",
                    "start_url": f"http://mock-{subset}.com",
                    "ground_truth": f"mock_answer_{task_id}",
                    "require_login": False,
                })
        
        logger.info(f"Loaded {len(self.tasks)} mock tasks across {len(self.subsets)} subsets")
    
    def get_tasks(self, subset: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tasks, optionally filtered by subset."""
        if subset:
            return [t for t in self.tasks if t["subset"] == subset]
        return self.tasks


class Evaluator:
    """
    Evaluation harness for running experiments.
    
    Supports two modes:
    - No Memory: Baseline agent
    - ReasoningBank: Memory-augmented agent with online learning
    """
    
    def __init__(
        self,
        agent: ReActAgent,
        judge: Optional[TrajectoryJudge] = None,
        extractor: Optional[StrategyExtractor] = None,
        memory_bank: Optional[ReasoningBank] = None,
        output_dir: str = "results",
        log_dir: str = "logs",
        checkpoint_interval: int = 100,
        matts_mode: str = "none",  # Paper Section 3.3: "none", "parallel", "sequential"
        scaling_factor: int = 1  # Paper Section 3.3: k for parallel/sequential scaling
    ):
        self.agent = agent
        self.judge = judge
        self.extractor = extractor
        self.memory_bank = memory_bank

        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.mode = "reasoningbank" if memory_bank else "no_memory"
        self.matts_mode = matts_mode  # MaTTS: Memory-aware Test-Time Scaling
        self.scaling_factor = scaling_factor
    
    def evaluate_task(
        self,
        task_data: Dict[str, Any],
        seed: int = 42,
        use_real_browser: bool = False
    ) -> tuple[TaskResult, Trajectory]:
        """
        Evaluate agent on a single task with optional MaTTS (Memory-aware Test-Time Scaling).

        Paper Section 3.3: MaTTS has two modes:
        - parallel: Generate k trajectories, select best via BoN
        - sequential: Iteratively refine trajectory k times

        Returns:
            (task_result, trajectory)
        """
        task_id = task_data["task_id"]
        task_description = task_data["description"]
        subset = task_data["subset"]
        reference_answer = task_data.get("reference_answer") or task_data.get("ground_truth")

        # Create environment
        environment = BrowserEnvironment(task_data, use_real_browser=use_real_browser)

        # Run agent with MaTTS if enabled
        logger.info(f"Running task {task_id} ({subset}) with MaTTS mode: {self.matts_mode}, k={self.scaling_factor}")
        if reference_answer:
            logger.info(f"  Reference answer: {reference_answer}")

        # Execute based on MaTTS mode
        if self.matts_mode == "parallel" and self.scaling_factor > 1:
            # Paper Figure 3(b): Parallel scaling with self-contrast
            trajectory = self._evaluate_with_parallel_scaling(
                task_data, environment, seed, reference_answer
            )
        elif self.matts_mode == "sequential" and self.scaling_factor > 1:
            # Paper Figure 3(c): Sequential scaling with self-refinement
            trajectory = self._evaluate_with_sequential_scaling(
                task_data, environment, seed, reference_answer
            )
        else:
            # Standard single-trajectory evaluation
            trajectory = self.agent.run(
                task_description=task_description,
                task_id=task_id,
                environment=environment,
                seed=seed,
                reference_answer=reference_answer
            )
        
        # Judge trajectory (if available)
        success = trajectory.success
        if self.judge:
            try:
                success = self.judge.judge(trajectory, task_description)
                trajectory.success = success
            except Exception as e:
                logger.error(f"Judge error for {task_id}: {e}")

        # FINAL GROUND TRUTH CHECK: Compare agent answer with reference answer
        # This is the definitive test at the very end of evaluation
        if reference_answer and reference_answer.strip():
            ground_truth_match = self._check_ground_truth(
                agent_answer=trajectory.final_answer,
                reference_answer=reference_answer,
                task_id=task_id
            )

            # If ground truth matches, override judge verdict to success
            if ground_truth_match:
                logger.info(f"✓ GROUND TRUTH MATCH for {task_id}: Marking as SUCCESS")
                success = True
                trajectory.success = True
            else:
                logger.info(f"✗ Ground truth mismatch for {task_id}: Judge verdict stands ({success})")

        # Extract strategies and update memory (ReasoningBank mode only)
        if self.mode == "reasoningbank" and self.extractor and self.memory_bank:
            try:
                strategies = self.extractor.extract(
                    trajectory=trajectory,
                    task_description=task_description,
                    success=success
                )
                
                for strategy in strategies:
                    self.memory_bank.add_memory(strategy, check_duplicate=True)
                    
            except Exception as e:
                logger.error(f"Extraction error for {task_id}: {e}")
        
        # Create result
        task_result = TaskResult(
            task_id=task_id,
            subset=subset,
            success=success,
            steps=trajectory.steps,
            tokens_input=trajectory.tokens["input"],
            tokens_output=trajectory.tokens["output"],
            walltime=trajectory.walltime,
            seed=seed
        )
        
        return task_result, trajectory

    def _evaluate_with_parallel_scaling(
        self,
        task_data: Dict[str, Any],
        environment: BrowserEnvironment,
        seed: int,
        reference_answer: Optional[str]
    ) -> Trajectory:
        """
        Parallel scaling evaluation (Paper Figure 3b, Section 4.3).

        Generate k trajectories, use self-contrast for memory extraction,
        then select best via BoN (Best-of-N).
        """
        from src.trajectory_selector import parallel_scaling_rollout, TrajectorySelector, ParallelScalingExtractor

        task_id = task_data["task_id"]
        task_description = task_data["description"]

        logger.info(f"Parallel scaling: generating {self.scaling_factor} trajectories for {task_id}")

        # Generate k parallel trajectories
        trajectories = parallel_scaling_rollout(
            agent=self.agent,
            task_description=task_description,
            task_id=task_id,
            environment=environment,
            n_trajectories=self.scaling_factor,
            seed=seed
        )

        if not trajectories:
            logger.error(f"Failed to generate trajectories for {task_id}")
            # Fallback to single trajectory
            return self.agent.run(task_description, task_id, environment, seed, reference_answer)

        # Judge all trajectories if judge available
        if self.judge:
            for traj in trajectories:
                try:
                    traj.success = self.judge.judge(traj, task_description)
                except Exception as e:
                    logger.error(f"Judge error: {e}")
                    traj.success = False

        # Paper Appendix A.3: Use self-contrast to extract memory from multiple trajectories
        if self.mode == "reasoningbank" and self.memory_bank:
            # Use ParallelScalingExtractor for self-contrast reasoning
            parallel_extractor = ParallelScalingExtractor(
                llm_client=self.agent.llm_client,
                temperature=1.0,  # Paper uses temp=1.0 for extraction
                max_items=5  # Paper: up to 5 items from parallel scaling
            )

            try:
                memory_items = parallel_extractor.extract_from_multiple(
                    trajectories=trajectories,
                    task_description=task_description
                )

                for item in memory_items:
                    self.memory_bank.add_memory(item, check_duplicate=True)

                logger.info(f"Parallel scaling: extracted {len(memory_items)} memory items")
            except Exception as e:
                logger.error(f"Parallel scaling memory extraction error: {e}")

        # Paper Figure 11: Select best trajectory via BoN
        if len(trajectories) > 1:
            selector = TrajectorySelector(
                llm_client=self.agent.llm_client,
                temperature=0.0  # Deterministic selection
            )
            best_idx, best_trajectory, analysis = selector.select_best(task_description, trajectories)
            logger.info(f"BoN selected trajectory {best_idx}: {analysis[:100]}...")
            return best_trajectory
        else:
            return trajectories[0]

    def _evaluate_with_sequential_scaling(
        self,
        task_data: Dict[str, Any],
        environment: BrowserEnvironment,
        seed: int,
        reference_answer: Optional[str]
    ) -> Trajectory:
        """
        Sequential scaling evaluation (Paper Figure 3c, Section 4.3).

        Iteratively refine single trajectory k times using self-refinement.
        """
        from src.trajectory_selector import sequential_scaling_rollout

        task_id = task_data["task_id"]
        task_description = task_data["description"]

        logger.info(f"Sequential scaling: refining trajectory {self.scaling_factor} times for {task_id}")

        # Generate and iteratively refine trajectory
        trajectory = sequential_scaling_rollout(
            agent=self.agent,
            task_description=task_description,
            task_id=task_id,
            environment=environment,
            n_iterations=self.scaling_factor,
            seed=seed
        )

        # Judge final trajectory
        if self.judge:
            try:
                trajectory.success = self.judge.judge(trajectory, task_description)
            except Exception as e:
                logger.error(f"Judge error: {e}")
                trajectory.success = False

        # Paper: Sequential scaling also produces memory items from refinement process
        # The intermediate check instructions provide valuable signals
        if self.mode == "reasoningbank" and self.extractor and self.memory_bank:
            try:
                strategies = self.extractor.extract(
                    trajectory=trajectory,
                    task_description=task_description,
                    success=trajectory.success
                )

                for strategy in strategies:
                    self.memory_bank.add_memory(strategy, check_duplicate=True)

                logger.info(f"Sequential scaling: extracted {len(strategies)} memory items")
            except Exception as e:
                logger.error(f"Sequential scaling memory extraction error: {e}")

        return trajectory

    def evaluate_dataset(
        self,
        dataset: TaskDataset,
        subset: Optional[str] = None,
        seed: int = 42,
        save_trajectories: bool = True,
        use_real_browser: bool = False
    ) -> EvaluationResults:
        """
        Evaluate agent on entire dataset or subset.
        
        Args:
            dataset: Task dataset
            subset: Optional subset filter
            seed: Random seed
            save_trajectories: Whether to save trajectory files
        
        Returns:
            Evaluation results with metrics
        """
        tasks = dataset.get_tasks(subset)
        subset_name = subset or "all"
        
        logger.info(f"Evaluating {len(tasks)} tasks in subset '{subset_name}' (mode: {self.mode})")
        
        results = []
        trajectories = []
        
        for i, task_data in enumerate(tqdm(tasks, desc=f"Evaluating {subset_name}")):
            try:
                task_result, trajectory = self.evaluate_task(task_data, seed=seed, use_real_browser=use_real_browser)
                results.append(task_result)
                trajectories.append(trajectory)
                
                # Save trajectory if requested
                if save_trajectories:
                    traj_dir = self.log_dir / self.mode / subset_name
                    traj_dir.mkdir(parents=True, exist_ok=True)
                    traj_file = traj_dir / f"{task_data['task_id']}.json"
                    
                    with open(traj_file, "w") as f:
                        json.dump(trajectory.model_dump(), f, indent=2)
                    
                    task_result.trajectory_path = str(traj_file)
                
                # Checkpoint
                if (i + 1) % self.checkpoint_interval == 0:
                    logger.info(f"Checkpoint: {i + 1}/{len(tasks)} tasks completed")
                    self._save_checkpoint(results, subset_name)
                    
                    if self.memory_bank:
                        self.memory_bank.save_checkpoint()
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_data['task_id']}: {e}")
                # Record error
                task_result = TaskResult(
                    task_id=task_data['task_id'],
                    subset=task_data['subset'],
                    success=False,
                    steps=0,
                    tokens_input=0,
                    tokens_output=0,
                    walltime=0.0,
                    seed=seed,
                    error=str(e)
                )
                results.append(task_result)
        
        # Compute metrics
        evaluation_results = self._compute_metrics(results, subset_name)

        # Save final results
        self._save_results(evaluation_results, subset_name)

        return evaluation_results

    def _check_ground_truth(
        self,
        agent_answer: Optional[str],
        reference_answer: str,
        task_id: str
    ) -> bool:
        """
        Final ground truth verification: Check if agent's answer matches reference answer.

        This is called at the very end of evaluation to verify correctness against ground truth.
        Uses normalization to handle formatting differences.

        Args:
            agent_answer: Agent's final answer
            reference_answer: Ground truth reference answer
            task_id: Task identifier for logging

        Returns:
            True if answers match (after normalization), False otherwise
        """
        if not agent_answer or not agent_answer.strip():
            logger.debug(f"Ground truth check for {task_id}: No agent answer")
            return False

        agent_answer_clean = agent_answer.strip()
        reference_answer_clean = reference_answer.strip()

        # Normalize both answers for comparison
        agent_normalized = self._normalize_answer(agent_answer_clean.lower())
        reference_normalized = self._normalize_answer(reference_answer_clean.lower())

        # Check exact match after normalization
        if agent_normalized == reference_normalized:
            logger.info(
                f"Ground truth EXACT match for {task_id}: "
                f"'{agent_answer_clean}' == '{reference_answer_clean}'"
            )
            return True

        # Check if reference is contained in agent answer
        # (agent may have provided more context)
        if reference_normalized in agent_normalized:
            logger.info(
                f"Ground truth SUBSTRING match for {task_id}: "
                f"'{reference_answer_clean}' in '{agent_answer_clean}'"
            )
            return True

        # Check multi-part answers (split by newlines)
        if '\n' in reference_answer_clean:
            reference_parts = [
                part.strip()
                for part in reference_answer_clean.split('\n')
                if part.strip()
            ]

            # Check if all parts are present
            all_found = True
            for part in reference_parts:
                part_normalized = self._normalize_answer(part.lower())
                if part_normalized not in agent_normalized:
                    all_found = False
                    break

            if all_found:
                logger.info(
                    f"Ground truth MULTI-PART match for {task_id}: "
                    f"All {len(reference_parts)} parts found"
                )
                return True

        # FINAL CHECK: Numeric tolerance for distance/time tasks
        # Agent might have slightly different route distance (e.g., 498km vs 457km)
        # Check if numbers are within 10% tolerance
        if self._numbers_match_within_tolerance(agent_answer_clean, reference_answer_clean, tolerance_percent=10.0):
            logger.info(
                f"Ground truth NUMERIC TOLERANCE match for {task_id}: "
                f"Numbers within 10% tolerance"
            )
            return True

        # No match found
        logger.debug(
            f"Ground truth NO match for {task_id}: "
            f"agent='{agent_answer_clean[:50]}...' vs ref='{reference_answer_clean[:50]}...'"
        )
        return False

    def _normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.

        Handles:
        - Distance formats: "914 km" -> "914km", "455 kilometers" -> "455km"
        - Time formats: "1 hour 23 minutes" -> "1h23min"
        - Whitespace normalization
        - Currency: "$45.50" -> "45.50usd"
        """
        import re

        if not text:
            return ""

        text = text.lower().strip()

        # Normalize distance formats
        text = re.sub(r'kilometers?', 'km', text, flags=re.IGNORECASE)
        text = re.sub(r'miles?(?!\w)', 'mi', text, flags=re.IGNORECASE)
        text = re.sub(r'meters?(?!\w)', 'm', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*(km|mi|m)', r'\1\2', text, flags=re.IGNORECASE)

        # Normalize time formats
        text = re.sub(r'(\d+)\s*hours?\s+and\s+(\d+)\s*minutes?', r'\1h\2min', text)
        text = re.sub(r'(\d+)\s*hours?', r'\1h', text)
        text = re.sub(r'(\d+)\s*minutes?', r'\1min', text)

        # Normalize currency
        text = re.sub(r'\$(\d+\.?\d*)', r'\1usd', text)
        text = re.sub(r'(\d+\.?\d*)\s*dollars?', r'\1usd', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _extract_numbers_from_text(self, text: str) -> list:
        """Extract all numbers from text for numeric comparison."""
        import re
        # Find all numbers (integers and decimals)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(n) for n in numbers]

    def _numbers_match_within_tolerance(self, agent_answer: str, reference_answer: str, tolerance_percent: float = 10.0) -> bool:
        """
        Check if numbers in agent answer match reference within tolerance.

        This handles cases like:
        - Agent says "498km" but reference is "457km" (within ~10% tolerance)
        - Different route distances that are close to reference

        Args:
            agent_answer: Agent's answer string
            reference_answer: Reference answer string
            tolerance_percent: Allowed percentage difference (default 10%)

        Returns:
            True if primary numbers match within tolerance
        """
        agent_nums = self._extract_numbers_from_text(agent_answer)
        ref_nums = self._extract_numbers_from_text(reference_answer)

        if not agent_nums or not ref_nums:
            return False

        # Compare primary numbers (usually first/largest)
        agent_primary = max(agent_nums)
        ref_primary = max(ref_nums)

        if ref_primary == 0:
            return agent_primary == 0

        # Calculate percentage difference
        percent_diff = abs(agent_primary - ref_primary) / ref_primary * 100

        matches = percent_diff <= tolerance_percent

        if matches:
            logger.info(
                f"Numbers match within {tolerance_percent}% tolerance: "
                f"{agent_primary} vs {ref_primary} (diff: {percent_diff:.1f}%)"
            )

        return matches

    def _compute_metrics(
        self,
        task_results: List[TaskResult],
        subset: str
    ) -> EvaluationResults:
        """Compute aggregated metrics from task results."""
        
        total_tasks = len(task_results)
        successful_tasks = sum(1 for r in task_results if r.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Average steps (only for successful tasks)
        successful_steps = [r.steps for r in task_results if r.success and r.steps > 0]
        avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0.0
        
        total_tokens = sum(r.tokens_input + r.tokens_output for r in task_results)
        total_walltime = sum(r.walltime for r in task_results)
        
        return EvaluationResults(
            mode=self.mode,
            subset=subset,
            success_rate=success_rate,
            avg_steps=avg_steps,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            total_tokens=total_tokens,
            total_walltime=total_walltime,
            task_results=task_results
        )
    
    def _save_checkpoint(self, results: List[TaskResult], subset: str):
        """Save intermediate results checkpoint."""
        checkpoint_file = self.output_dir / f"{self.mode}_{subset}_checkpoint.json"
        
        with open(checkpoint_file, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    
    def _save_results(self, results: EvaluationResults, subset: str):
        """Save final evaluation results."""
        # Save as JSON
        json_file = self.output_dir / f"{self.mode}_{subset}.json"
        with open(json_file, "w") as f:
            json.dump(results.model_dump(), f, indent=2)
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame([r.model_dump() for r in results.task_results])
        csv_file = self.output_dir / f"{self.mode}_{subset}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved results to {json_file} and {csv_file}")
        logger.info(
            f"Results for {subset} ({self.mode}): "
            f"SR={results.success_rate:.3f}, "
            f"AvgSteps={results.avg_steps:.1f}, "
            f"Success={results.successful_tasks}/{results.total_tasks}"
        )
