"""
Trajectory Selection for Best-of-N (BoN) and MATTS.

This module implements:
- Figure 11: Best-of-N trajectory selection
- Figure 10: MaTTS (Memory-Aware Test-Time Scaling) with parallel and sequential scaling
"""

from typing import List, Optional
import uuid
from datetime import datetime
from loguru import logger

from src.models import Trajectory, MemoryItem
from src.llm_client import LLMClient


# Best-of-N Selection Prompt (Figure 11)
BON_TRAJECTORY_SELECTOR_PROMPT = """You are an expert in evaluating web navigation agent trajectories. You will be given the user query, and {N} candidate trajectories, each representing a sequence of steps for solving the same task. Your job is to select the single best trajectory that most effectively and efficiently solves the task, and explain your reasoning.

## Input Format:
Each trajectory consists of multiple steps. For each step, you will be provided:
- step_num: Step index in the trajectory.
- action_output: The action the agent takes (click, type, scroll, etc.).
- think_output: The agent's reasoning or plan before taking the action.

## Evaluation Criteria:

### Progress Toward Goal
1. How well the trajectory advances toward completing the user's task.
2. Reward tangible, meaningful progress; penalize minimal or no advancement.
3. Consider both individual step contributions and overall progress.

### Trajectory Efficiency
1. How efficiently the trajectory achieves progress given the number and complexity of steps.
2. Reward significant progress in fewer steps.
3. Favor better value-to-depth ratios.
4. Reward efficient search space exploration.

### Loop Detection
Identify loops or redundant actions.
1. Real Loops: Repeating identical observations and actions with no added value.
2. Benign Repetitions: Slight variations that still yield new information.
3. Penalize real loops heavily; penalize benign repetitions only if they waste effort.

### Error Severity and Stability
Assess severity of errors:
1. Fatal/Blocking: Major penalty.
2. Significant: Moderate penalty.
3. Minor/Recoverable: Minor penalty.
4. Penalize unstable or incoherent model reasoning.
5. Consider whether errors prevent goal completion.

### Overall Trajectory Quality
1. Logical flow of steps, clarity of strategy, and coherence.
2. Balanced exploration vs. exploitation.
3. Closeness to final goal.
4. Reward consistent progress and coherent planning.

## Output Format:
Return the evaluation as a JSON object:
```{"index": [best_trajectory_index], "analysis": "Detailed reasoning explaining why this trajectory is the best, referencing progress, efficiency, loop detection, error severity, and overall quality."}```"""


# MaTTS Prompts from paper Appendix A.3 (Figure 10)

# Parallel Scaling: Extract memory from multiple trajectories using self-contrast
PARALLEL_SCALING_SYSTEM_PROMPT = """You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task. Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory items.

Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
- Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
- Think first: Why did some trajectories succeed while others failed?
- You can extract at most 5 memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
```"""


# Sequential Scaling: Check instructions for iterative refinement
FIRST_TIME_CHECK_INSTRUCTION = """Important: Let's carefully re-examine the previous trajectory, including your reasoning steps and actions taken.

Pay special attention to whether you used the correct elements on the page, and whether your response addresses the user query. If you find inconsistencies, correct them. If everything seems correct, confirm your final answer.

Output must stay in the same "<think>...</think><action></action>" format as previous trajectories."""


FOLLOWUP_CHECK_INSTRUCTION = """Let's check again.

Output must stay in the same "<think>...</think><action></action>" format as previous trajectories."""


class TrajectorySelector:
    """
    Selects the best trajectory from multiple candidates using LLM-as-Judge.
    
    This implements the Best-of-N (BoN) selection from the paper's MATTS approach.
    """
    
    def __init__(self, llm_client: LLMClient, temperature: float = 0.0):
        self.llm_client = llm_client
        self.temperature = temperature
    
    def select_best(
        self,
        query: str,
        trajectories: List[Trajectory]
    ) -> tuple[int, Trajectory, str]:
        """
        Select the best trajectory from N candidates using Figure 11 prompt.

        Args:
            query: Original task query/description
            trajectories: List of N candidate trajectories

        Returns:
            (best_index, best_trajectory, analysis)
        """
        if len(trajectories) == 1:
            return 0, trajectories[0], "Only one trajectory available"

        # Format trajectories for comparison
        trajectory_texts = []
        for i, traj in enumerate(trajectories, 1):
            traj_steps = []
            for step_num, (action, thought) in enumerate(zip(traj.actions, traj.thoughts), 1):
                traj_steps.append(
                    f"  Step {step_num}:\n"
                    f"    think_output: {thought}\n"
                    f"    action_output: {action.tool}({action.args})"
                )
            trajectory_texts.append(f"Trajectory {i}:\n" + "\n".join(traj_steps[:15]))  # Limit steps

        # Build user prompt with trajectories
        user_prompt = f"""Query: {query}

{chr(10).join(trajectory_texts)}"""

        # Get LLM selection
        messages = [
            {"role": "system", "content": BON_TRAJECTORY_SELECTOR_PROMPT.format(N=len(trajectories))},
            {"role": "user", "content": user_prompt}
        ]

        response, _ = self.llm_client.complete(
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000
        )

        # Parse response (expects JSON with "index" and "analysis")
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                best_idx = int(result.get("index", 0))
                analysis = result.get("analysis", "No analysis provided")

                # Validate index
                if 0 <= best_idx < len(trajectories):
                    return best_idx, trajectories[best_idx], analysis
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: return first trajectory
        return 0, trajectories[0], "Failed to parse selection, using first trajectory"


class ParallelScalingExtractor:
    """
    Extracts memory from multiple trajectories using self-contrast reasoning.
    Implements Figure 10 (left panel) - Parallel Scaling.
    """

    def __init__(self, llm_client: LLMClient, temperature: float = 1.0, max_items: int = 5):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_items = max_items

    def extract_from_multiple(
        self,
        trajectories: List[Trajectory],
        task_description: str
    ) -> List[MemoryItem]:
        """
        Extract memory items from multiple trajectories using self-contrast.

        Args:
            trajectories: List of trajectories (mix of successful and failed)
            task_description: Original task description

        Returns:
            List of extracted memory items (up to max_items=5)
        """
        if not trajectories:
            return []

        # Format all trajectories
        trajectory_texts = []
        for i, traj in enumerate(trajectories, 1):
            traj_steps = []
            for thought, action in zip(traj.thoughts, traj.actions):
                traj_steps.append(f"<think>{thought}</think>\n<action>{action.tool}</action>")

            trajectory_text = "\n".join(traj_steps[:20])
            if len(traj.actions) > 20:
                trajectory_text += f"\n... ({len(traj.actions) - 20} more steps)"

            trajectory_texts.append(f"Trajectory {i}:\n{trajectory_text}")

        # Build user prompt
        user_prompt = f"""Query: {task_description}

Trajectories: {chr(10).join(trajectory_texts)}"""

        messages = [
            {"role": "system", "content": PARALLEL_SCALING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response, tokens = self.llm_client.complete(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048
            )

            if response.startswith("Error:"):
                logger.warning(f"Parallel scaling extractor LLM error: {response}")
                return []

            # Parse Markdown response
            response = response.strip()
            if not response:
                logger.warning("Empty response from parallel scaling extractor")
                return []

            # Parse Markdown format memory items (reuse parsing logic)
            memory_items = self._parse_markdown_memories(response, trajectories)

            logger.info(f"Extracted {len(memory_items)} strategies from {len(trajectories)} trajectories")

            return memory_items[:self.max_items]

        except Exception as e:
            logger.error(f"Parallel scaling extraction error: {e}")
            return []

    def _parse_markdown_memories(
        self,
        response: str,
        trajectories: List[Trajectory]
    ) -> List[MemoryItem]:
        """Parse Markdown formatted memory items from LLM response."""
        memory_items = []

        # Split by "# Memory Item" markers
        import re
        items = re.split(r'#\s+Memory\s+Item\s+\d+|#\s+Memory\s+Item\s+i', response, flags=re.IGNORECASE)

        # Skip first split (text before first item)
        for item_text in items[1:]:
            if not item_text.strip():
                continue

            try:
                # Extract title
                title_match = re.search(r'##\s+Title\s+(.+?)(?=\n|$)', item_text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "Untitled Strategy"

                # Extract description
                desc_match = re.search(r'##\s+Description\s+(.+?)(?=\n##|\n#|$)', item_text, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""

                # Extract content
                content_match = re.search(r'##\s+Content\s+(.+?)(?=\n##|\n#|$)', item_text, re.IGNORECASE | re.DOTALL)
                content_text = content_match.group(1).strip() if content_match else ""

                # Convert content to list
                if content_text:
                    content = [c.strip('- ').strip() for c in content_text.split('\n') if c.strip()]
                    if not content:
                        content = [content_text]
                else:
                    content = []

                # Create memory item
                memory_item = MemoryItem(
                    id=str(uuid.uuid4()),
                    title=title,
                    description=description,
                    content=content,
                    provenance={
                        "source": "parallel_scaling",
                        "num_trajectories": len(trajectories),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                memory_items.append(memory_item)

            except Exception as e:
                logger.warning(f"Failed to parse memory item: {e}")
                continue

        return memory_items


def parallel_scaling_rollout(
    agent,
    task_description: str,
    task_id: str,
    environment,
    n_trajectories: int = 5,
    seed: int = 42
) -> List[Trajectory]:
    """
    Parallel scaling: Generate N independent trajectories for the same task.

    This implements the "parallel scaling" component of MATTS from the paper.

    Args:
        agent: ReActAgent instance
        task_description: Task description
        task_id: Task identifier
        environment: Browser environment
        n_trajectories: Number of trajectories to generate
        seed: Base random seed

    Returns:
        List of N trajectories
    """
    import random
    import numpy as np

    logger.info(f"Starting parallel scaling rollout: {n_trajectories} trajectories for task {task_id}")

    trajectories = []
    for i in range(n_trajectories):
        # Use different seed for each trajectory to encourage variation
        trajectory_seed = seed + i
        random.seed(trajectory_seed)
        np.random.seed(trajectory_seed)

        logger.info(f"Generating trajectory {i+1}/{n_trajectories} with seed {trajectory_seed}")

        try:
            # Reset environment for fresh trajectory
            environment.reset()

            # Run agent to generate trajectory
            # Agent should use higher temperature for variation
            trajectory = agent.run(
                task_description=task_description,
                task_id=f"{task_id}_parallel_{i}",
                environment=environment
            )

            trajectories.append(trajectory)
            logger.debug(f"Trajectory {i+1} completed with {len(trajectory.actions)} steps")

        except Exception as e:
            logger.error(f"Failed to generate trajectory {i+1}: {e}")
            # Continue with remaining trajectories
            continue

    logger.info(f"Parallel scaling completed: {len(trajectories)}/{n_trajectories} trajectories generated")
    return trajectories


def sequential_scaling_rollout(
    agent,
    task_description: str,
    task_id: str,
    environment,
    n_iterations: int = 3,
    seed: int = 42
) -> Trajectory:
    """
    Sequential scaling: Iteratively refine a single trajectory.

    This implements the "sequential scaling" component of MATTS from the paper.
    The agent repeatedly re-examines its own trajectory with check instructions.

    Args:
        agent: ReActAgent instance
        task_description: Task description
        task_id: Task identifier
        environment: Browser environment
        n_iterations: Number of refinement iterations
        seed: Random seed

    Returns:
        Final refined trajectory
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Starting sequential scaling rollout: {n_iterations} iterations for task {task_id}")

    # Generate initial trajectory
    logger.info("Generating initial trajectory")
    try:
        environment.reset()
        trajectory = agent.run(
            task_description=task_description,
            task_id=f"{task_id}_sequential_0",
            environment=environment
        )
        logger.debug(f"Initial trajectory completed with {len(trajectory.actions)} steps")
    except Exception as e:
        logger.error(f"Failed to generate initial trajectory: {e}")
        raise

    # Iteratively refine the trajectory
    for iteration in range(n_iterations):
        # Choose appropriate check instruction
        if iteration == 0:
            check_instruction = FIRST_TIME_CHECK_INSTRUCTION
            logger.info(f"Refining trajectory: iteration {iteration+1}/{n_iterations} (first-time check)")
        else:
            check_instruction = FOLLOWUP_CHECK_INSTRUCTION
            logger.info(f"Refining trajectory: iteration {iteration+1}/{n_iterations} (follow-up check)")

        try:
            # Reset environment to same state
            environment.reset()

            # Extend task description with check instruction
            extended_task = f"{task_description}\n\n{check_instruction}"

            # Run agent with extended instructions
            # Agent should review previous trajectory and refine
            refined_trajectory = agent.run(
                task_description=extended_task,
                task_id=f"{task_id}_sequential_{iteration+1}",
                environment=environment,
                previous_trajectory=trajectory  # Pass previous trajectory for refinement
            )

            # Update trajectory
            trajectory = refined_trajectory
            logger.debug(f"Refinement {iteration+1} completed with {len(trajectory.actions)} steps")

        except Exception as e:
            logger.error(f"Failed to refine trajectory at iteration {iteration+1}: {e}")
            # Return best trajectory so far
            break

    logger.info(f"Sequential scaling completed with {len(trajectory.actions)} final steps")
    return trajectory

