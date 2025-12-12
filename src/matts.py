"""
Memory-Aware Test-Time Scaling (MaTTS) Implementation.

This module provides the complete MaTTS framework from the ReasoningBank paper:
- Parallel Scaling: Generate N trajectories and extract contrastive insights
- Sequential Scaling: Iteratively refine a trajectory with self-checks
- Integrated workflow combining scaling with memory extraction

Paper Reference: Section 3.3, Figures 3, 4, and 10
"""

import random
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from src.models import Trajectory, MemoryItem
from src.memory import ReasoningBank
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.trajectory_selector import (
    TrajectorySelector,
    ParallelScalingExtractor,
    parallel_scaling_rollout,
    sequential_scaling_rollout
)
from src.llm_client import LLMClient


class MaTTS:
    """
    Memory-Aware Test-Time Scaling (MaTTS) Framework.

    Implements both parallel and sequential scaling modes with integrated
    memory extraction and consolidation.

    Usage:
        matts = MaTTS(llm_client, reasoning_bank, agent, environment)

        # Parallel mode (Best-of-N)
        best_traj, memories = matts.run_parallel(
            task_description="Find...",
            task_id="task_1",
            k=5  # Generate 5 trajectories
        )

        # Sequential mode (iterative refinement)
        refined_traj, memories = matts.run_sequential(
            task_description="Find...",
            task_id="task_2",
            k=3  # 3 refinement iterations
        )
    """

    def __init__(
        self,
        llm_client: LLMClient,
        reasoning_bank: ReasoningBank,
        agent,
        environment,
        temperature_judge: float = 0.0,
        temperature_extract: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize MaTTS framework.

        Args:
            llm_client: LLM client for completions
            reasoning_bank: ReasoningBank instance for memory storage
            agent: Agent instance with run() method
            environment: Browser/environment instance
            temperature_judge: Temperature for judge/selector (0.0 for deterministic)
            temperature_extract: Temperature for extraction (1.0 for diversity)
            seed: Base random seed
        """
        self.llm_client = llm_client
        self.reasoning_bank = reasoning_bank
        self.agent = agent
        self.environment = environment
        self.seed = seed

        # Initialize components
        self.judge = TrajectoryJudge(llm_client, temperature=temperature_judge)
        self.extractor = StrategyExtractor(llm_client, temperature=temperature_extract)
        self.selector = TrajectorySelector(llm_client, temperature=temperature_judge)
        self.parallel_extractor = ParallelScalingExtractor(
            llm_client,
            temperature=temperature_extract,
            max_items=5
        )

    def run_parallel(
        self,
        task_description: str,
        task_id: str,
        k: int = 5,
        extract_mode: str = "self_contrast"
    ) -> Tuple[Trajectory, List[MemoryItem]]:
        """
        Run parallel scaling (Figure 3b and 10 left panel).

        Process:
        1. Generate k independent trajectories
        2. Select best trajectory using Best-of-N
        3. Extract memories using self-contrast across all trajectories
        4. Return best trajectory and extracted memories

        Args:
            task_description: Task description
            task_id: Task identifier
            k: Number of trajectories to generate (scaling factor)
            extract_mode: "self_contrast" (compare all) or "individual" (extract per trajectory)

        Returns:
            (best_trajectory, memory_items)
        """
        logger.info(f"MaTTS Parallel Scaling: k={k}, task={task_id}")

        # Step 1: Generate k trajectories in parallel
        logger.info(f"Generating {k} independent trajectories...")
        trajectories = parallel_scaling_rollout(
            agent=self.agent,
            task_description=task_description,
            task_id=task_id,
            environment=self.environment,
            n_trajectories=k,
            seed=self.seed
        )

        if not trajectories:
            logger.error("No trajectories generated!")
            raise ValueError("Parallel scaling failed to generate any trajectories")

        logger.info(f"Generated {len(trajectories)} trajectories")

        # Step 2: Select best trajectory using Best-of-N (Figure 11)
        logger.info("Selecting best trajectory using Best-of-N...")
        best_idx, best_trajectory, analysis = self.selector.select_best(
            query=task_description,
            trajectories=trajectories
        )
        logger.info(f"Best trajectory: index={best_idx}, steps={best_trajectory.steps}")
        logger.debug(f"Selection analysis: {analysis}")

        # Step 3: Extract memories using self-contrast (Figure 10 left)
        logger.info(f"Extracting memories using mode: {extract_mode}")

        if extract_mode == "self_contrast":
            # Use self-contrast extractor for all trajectories together
            memory_items = self.parallel_extractor.extract_from_multiple(
                trajectories=trajectories,
                task_description=task_description
            )
            logger.info(f"Self-contrast extracted {len(memory_items)} memory items")

        elif extract_mode == "individual":
            # Extract from each trajectory individually (Vanilla TTS)
            memory_items = []
            for i, traj in enumerate(trajectories):
                # Judge each trajectory
                success = self.judge.judge(traj, task_description)
                # Extract memories
                items = self.extractor.extract(traj, task_description, success)
                memory_items.extend(items)
                logger.debug(f"Trajectory {i}: {len(items)} items (success={success})")

            logger.info(f"Individual extraction: {len(memory_items)} total items")

        else:
            raise ValueError(f"Unknown extract_mode: {extract_mode}")

        # Step 4: Add memories to ReasoningBank
        for item in memory_items:
            self.reasoning_bank.add_memory(item, check_duplicate=True)

        logger.info(f"MaTTS Parallel completed: {len(memory_items)} memories added")
        return best_trajectory, memory_items

    def run_sequential(
        self,
        task_description: str,
        task_id: str,
        k: int = 3
    ) -> Tuple[Trajectory, List[MemoryItem]]:
        """
        Run sequential scaling (Figure 3c and 10 right panel).

        Process:
        1. Generate initial trajectory
        2. Iteratively refine k times with check instructions
        3. Extract memories from final refined trajectory
        4. Return refined trajectory and extracted memories

        Args:
            task_description: Task description
            task_id: Task identifier
            k: Number of refinement iterations (scaling factor)

        Returns:
            (refined_trajectory, memory_items)
        """
        logger.info(f"MaTTS Sequential Scaling: k={k}, task={task_id}")

        # Step 1-2: Generate and refine trajectory
        logger.info(f"Running sequential refinement with {k} iterations...")
        refined_trajectory = sequential_scaling_rollout(
            agent=self.agent,
            task_description=task_description,
            task_id=task_id,
            environment=self.environment,
            n_iterations=k,
            seed=self.seed
        )

        logger.info(f"Refinement completed: {refined_trajectory.steps} steps")

        # Step 3: Extract memories from refined trajectory
        logger.info("Extracting memories from refined trajectory...")
        success = self.judge.judge(refined_trajectory, task_description)
        memory_items = self.extractor.extract(
            trajectory=refined_trajectory,
            task_description=task_description,
            success=success
        )

        logger.info(f"Extracted {len(memory_items)} memory items (success={success})")

        # Step 4: Add memories to ReasoningBank
        for item in memory_items:
            self.reasoning_bank.add_memory(item, check_duplicate=True)

        logger.info(f"MaTTS Sequential completed: {len(memory_items)} memories added")
        return refined_trajectory, memory_items

    def run_vanilla_tts(
        self,
        task_description: str,
        task_id: str,
        mode: str = "parallel",
        k: int = 5
    ) -> Tuple[Trajectory, List[MemoryItem]]:
        """
        Run Vanilla Test-Time Scaling without aggregation (Figure 3a).

        This is "MaTTS w/o aggregation" from paper experiments.
        Generates multiple trajectories but extracts memories individually.

        Args:
            task_description: Task description
            task_id: Task identifier
            mode: "parallel" or "sequential"
            k: Scaling factor

        Returns:
            (best_trajectory, memory_items)
        """
        logger.info(f"Vanilla TTS ({mode}): k={k}, task={task_id}")

        if mode == "parallel":
            # Use parallel mode with individual extraction
            return self.run_parallel(
                task_description=task_description,
                task_id=task_id,
                k=k,
                extract_mode="individual"  # No self-contrast
            )
        else:
            # Sequential mode (same as MaTTS sequential)
            return self.run_sequential(
                task_description=task_description,
                task_id=task_id,
                k=k
            )


def run_matts_evaluation(
    matts: MaTTS,
    tasks: List[dict],
    mode: str = "parallel",
    k: int = 5,
    output_dir: Optional[str] = None
) -> dict:
    """
    Run MaTTS evaluation on a list of tasks.

    Args:
        matts: MaTTS instance
        tasks: List of task dictionaries with 'id', 'description', etc.
        mode: "parallel", "sequential", or "vanilla"
        k: Scaling factor
        output_dir: Optional directory to save results

    Returns:
        Dictionary with evaluation results
    """
    import json
    from pathlib import Path
    from datetime import datetime

    logger.info(f"Running MaTTS evaluation: mode={mode}, k={k}, n_tasks={len(tasks)}")

    results = {
        "mode": mode,
        "scaling_factor": k,
        "timestamp": datetime.utcnow().isoformat(),
        "tasks": []
    }

    for i, task in enumerate(tasks, 1):
        logger.info(f"Task {i}/{len(tasks)}: {task['id']}")

        try:
            if mode == "parallel":
                trajectory, memories = matts.run_parallel(
                    task_description=task['description'],
                    task_id=task['id'],
                    k=k
                )
            elif mode == "sequential":
                trajectory, memories = matts.run_sequential(
                    task_description=task['description'],
                    task_id=task['id'],
                    k=k
                )
            elif mode == "vanilla":
                trajectory, memories = matts.run_vanilla_tts(
                    task_description=task['description'],
                    task_id=task['id'],
                    mode="parallel",
                    k=k
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            task_result = {
                "task_id": task['id'],
                "success": trajectory.success,
                "steps": trajectory.steps,
                "memories_extracted": len(memories),
                "final_answer": trajectory.final_answer
            }
            results["tasks"].append(task_result)

            logger.info(f"Task {task['id']}: success={trajectory.success}, steps={trajectory.steps}")

        except Exception as e:
            logger.error(f"Task {task['id']} failed: {e}")
            results["tasks"].append({
                "task_id": task['id'],
                "error": str(e)
            })

    # Compute summary statistics
    successful = [t for t in results["tasks"] if t.get("success", False)]
    results["summary"] = {
        "total_tasks": len(tasks),
        "successful_tasks": len(successful),
        "success_rate": len(successful) / len(tasks) if tasks else 0.0,
        "avg_steps": sum(t["steps"] for t in results["tasks"] if "steps" in t) / len(tasks) if tasks else 0.0
    }

    logger.info(f"Evaluation complete: SR={results['summary']['success_rate']:.2%}, "
                f"Avg Steps={results['summary']['avg_steps']:.1f}")

    # Save results
    if output_dir:
        output_path = Path(output_dir) / f"matts_{mode}_k{k}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results
