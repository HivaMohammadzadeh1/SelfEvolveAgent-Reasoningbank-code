"""LLM-as-Judge for trajectory success/failure classification."""
from typing import Dict, Any
from loguru import logger

from src.llm_client import LLMClient
from src.models import Trajectory


# Improved judge prompt to better detect failures
JUDGE_SYSTEM_PROMPT = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to
complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to
decide whether the agent's execution is successful or not. 

Carefully analyze the task requirements and the agent's execution.

FAILURE indicators:
- Agent gave up or said it cannot complete the task
- Agent reported errors, empty pages, or inability to interact
- Final answer is empty, generic, or doesn't address the task
- Agent repeated the same failed action multiple times
- Agent couldn't load pages or see page content
- Final answer contains phrases like "unable to", "cannot", "not loading", "error"

SUCCESS indicators:
- Agent provided a specific answer that addresses the task requirements
- Agent successfully navigated and interacted with the required pages
- Final answer contains concrete information relevant to the task

Respond with EXACTLY one word:
- "Success" if the task was clearly completed with a valid answer
- "Failure" if the task was not completed, had errors, or the answer is invalid

Output only ONE word, nothing else."""


def sanitize_text(text: str, max_length: int = 500) -> str:
    """Sanitize text to avoid triggering safety filters."""
    if not text:
        return ""

    # Remove URLs and domain names that might trigger filters
    import re
    # Replace full URLs
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    # Replace domain-like patterns
    text = re.sub(r'\b[a-z0-9-]+\.[a-z]{2,}\b', '[DOMAIN]', text, flags=re.IGNORECASE)
    # Replace IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def format_trajectory_for_judge(trajectory: Trajectory, task_description: str) -> str:
    """Format trajectory for judge evaluation."""

    # Sanitize task description
    safe_task = sanitize_text(task_description, max_length=300)

    # Summarize actions (tool names only, skip args to avoid sensitive data)
    action_summary = []
    for i, (action, thought) in enumerate(zip(trajectory.actions, trajectory.thoughts), 1):
        # Sanitize thought and action
        safe_thought = sanitize_text(thought, max_length=100)
        action_summary.append(
            f"Step {i}: {safe_thought} -> {action.tool}()"
        )

    trajectory_text = "\n".join(action_summary[:15])  # Limit to first 15 steps
    if len(trajectory.actions) > 15:
        trajectory_text += f"\n... ({len(trajectory.actions) - 15} more steps)"

    # Sanitize final answer
    safe_answer = sanitize_text(trajectory.final_answer or "No final answer provided", max_length=200)

    content = f"""Task: {safe_task}

Final Answer: {safe_answer}

Steps Taken: {trajectory.steps} actions
Trajectory Summary:
{trajectory_text}

Was the task completed successfully?"""

    return content


class TrajectoryJudge:
    """LLM-as-Judge for evaluating trajectory success/failure."""
    
    def __init__(self, llm_client: LLMClient, temperature: float = 0.0):
        self.llm_client = llm_client
        self.temperature = temperature
    
    def judge(self, trajectory: Trajectory, task_description: str) -> bool:
        """
        Evaluate if trajectory was successful.
        
        Args:
            trajectory: Agent trajectory to evaluate
            task_description: Original task description
        
        Returns:
            True if successful, False otherwise
        """
        # Heuristic checks for obvious failures (before calling LLM)
        if self._is_obvious_failure(trajectory):
            logger.debug(f"Judge: {trajectory.task_id} is obvious failure (heuristic)")
            return False
        
        # If no final answer was provided, it's a failure
        if not trajectory.final_answer or trajectory.final_answer.strip() == "":
            logger.debug(f"Judge: {trajectory.task_id} failed - no final answer")
            return False
        
        # Prepare content for LLM judge
        user_content = format_trajectory_for_judge(trajectory, task_description)
        
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response, tokens = self.llm_client.complete(
                messages=messages,
                temperature=self.temperature,
                max_tokens=50
            )
            
            # Check for errors (safety filters, etc.)
            if response.startswith("Error:"):
                logger.warning(f"Judge LLM error for {trajectory.task_id}: {response}")
                # When LLM fails, use heuristics to make a best guess
                # Default to failure unless there's clear evidence of success
                success = self._heuristic_judge(trajectory, task_description)
                logger.debug(f"Using heuristic fallback: {success}")
                return success
            
            # Parse response
            response_lower = response.strip().lower()
            success = "success" in response_lower and "failure" not in response_lower
            
            logger.debug(
                f"Judge verdict for {trajectory.task_id}: "
                f"{'Success' if success else 'Failure'} (raw: {response[:50]})"
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Judge error for {trajectory.task_id}: {e}")
            # When judge fails, use heuristics
            success = self._heuristic_judge(trajectory, task_description)
            logger.debug(f"Judge failed, using heuristic: {success}")
            return success
    
    def _is_obvious_failure(self, trajectory: Trajectory) -> bool:
        """Check for obvious failure indicators."""
        if not trajectory.final_answer:
            return True
        
        final_answer_lower = trajectory.final_answer.lower()
        
        # Failure keywords in answer
        failure_phrases = [
            "unable to", "cannot", "can't", "could not", "couldn't",
            "not loading", "not displaying", "empty", "error",
            "failed to", "impossible", "not working", "not available",
            "accessibility tree is empty", "no content", "page is not loading",
            "preventing me", "preventing interaction", "no elements",
            "consistently returns", "i am unable", "i cannot",
            "does not load", "not responding", "no page content",
            "empty page", "blank page", "nothing to interact",
            "no observable", "returns empty", "returns []"
        ]
        
        for phrase in failure_phrases:
            if phrase in final_answer_lower:
                return True
        
        # Very short answers are usually failures (less than 10 chars)
        if len(trajectory.final_answer.strip()) < 10:
            return True
        
        # Agent took very few steps (less than 2)
        if trajectory.steps < 2:
            return True
        
        # Check if agent navigated to external sites (shouldn't happen, but indicates failure)
        external_domains = ["google.com", "wikipedia.org", "youtube.com", "facebook.com", "twitter.com"]
        for action in trajectory.actions:
            if action.tool == "navigate":
                url = action.args.get("url", "").lower()
                for domain in external_domains:
                    if domain in url and "ec2" not in url:
                        logger.debug(f"Detected external navigation to {domain} - marking as failure")
                        return True
        
        return False
    
    def _heuristic_judge(self, trajectory: Trajectory, task_description: str) -> bool:
        """Fallback heuristic judgment when LLM judge fails."""
        # Check for obvious failure
        if self._is_obvious_failure(trajectory):
            return False
        
        # Check if answer seems substantial (more than 20 characters, not error message)
        if trajectory.final_answer:
            answer = trajectory.final_answer.strip()
            if len(answer) > 20 and not any(word in answer.lower() for word in ["error", "unable", "cannot", "failed"]):
                # Seems like a real attempt at an answer
                # Conservative: still default to False unless we're confident
                return False  # Be conservative - most likely still a failure
        
        return False  # Default to failure when uncertain
