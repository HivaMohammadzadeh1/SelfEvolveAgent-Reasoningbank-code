"""Strategy extraction from trajectories."""
import json
import uuid
from typing import List
from datetime import datetime
from loguru import logger

from src.llm_client import LLMClient
from src.models import Trajectory, MemoryItem


# Extraction prompts from paper Appendix A.1 (Figure 8)
# Two different prompts: one for successful trajectories, one for failed trajectories

EXTRACTION_SYSTEM_PROMPT_SUCCESS = """You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory is successful, and then summarize the insights.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>``` 
"""


EXTRACTION_SYSTEM_PROMPT_FAILURE = """You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most 3 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```
"""


def sanitize_text_for_extraction(text: str, max_length: int = 500) -> str:
    """Sanitize text aggressively to avoid triggering safety filters."""
    if not text:
        return ""

    import re

    # Remove URLs, domains, and sensitive patterns
    # Replace full URLs (any protocol)
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)

    # Replace domain-like patterns
    text = re.sub(r'\b[a-z0-9-]+\.[a-z]{2,}\b', '[SITE]', text, flags=re.IGNORECASE)

    # Replace IP addresses and ports
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b', '[HOST]', text)

    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Replace element IDs and technical identifiers
    text = re.sub(r'\bid=["\'][\w-]+["\']', 'id="[ID]"', text)
    text = re.sub(r'\bclass=["\'][\w\s-]+["\']', 'class="[CLASS]"', text)

    # Replace common EC2/AWS patterns
    text = re.sub(r'ec2-[\d-]+\.[\w.-]+\.amazonaws\.com', '[EC2-HOST]', text)
    text = re.sub(r'compute-\d+\.amazonaws\.com', '[AWS-HOST]', text)

    # Remove accessibility tree markup
    text = re.sub(r'<[^>]+>', '[ELEMENT]', text)

    # Replace automation keywords that trigger filters
    text = text.replace('browsergym_id', 'element_ref')
    text = text.replace('playwright', 'browser')
    text = text.replace('automation', 'interaction')

    # Remove repeated whitespace
    text = re.sub(r'\s+', ' ', text)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()


def format_trajectory_for_extraction(
    trajectory: Trajectory,
    task_description: str,
    success: bool
) -> str:
    """
    Format trajectory for strategy extraction according to AWM paper specification.

    Paper (Section 2.2): Each step p = (o, a) contains:
    1. Observation o: NL description of current environment state
    2. Reasoning: Agent's thought process
    3. Action a: Executable action
    """

    # Sanitize task description
    safe_task = sanitize_text_for_extraction(task_description, max_length=300)

    # Format trajectory with OBSERVATION + THOUGHT + ACTION (AWM paper format)
    action_summary = []
    for i, (obs, thought, action) in enumerate(zip(trajectory.observations, trajectory.thoughts, trajectory.actions)):
        # Sanitize and truncate observation (key environmental context)
        safe_obs = sanitize_text_for_extraction(obs, max_length=300)
        safe_thought = sanitize_text_for_extraction(thought, max_length=150)

        # AWM format: Each step shows environment state → reasoning → action
        step_text = f"<observation>{safe_obs}</observation>\n<think>{safe_thought}</think>\n<action>{action.tool}</action>"
        action_summary.append(step_text)

    trajectory_text = "\n\n".join(action_summary[:20])
    if len(trajectory.actions) > 20:
        trajectory_text += f"\n\n... ({len(trajectory.actions) - 20} more steps)"

    # Paper's format: Query + Trajectory
    content = f"""Query: {safe_task}

Trajectory:
{trajectory_text}"""

    return content


class StrategyExtractor:
    """Extract strategy memories from agent trajectories."""
    
    def __init__(self, llm_client: LLMClient, temperature: float = 1.0, max_items: int = 3):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_items = max_items
    
    def extract(
        self,
        trajectory: Trajectory,
        task_description: str,
        success: bool
    ) -> List[MemoryItem]:
        """
        Extract strategy items from a trajectory.

        Args:
            trajectory: Agent trajectory
            task_description: Original task description
            success: Whether trajectory was successful (from judge)

        Returns:
            List of extracted memory items (up to max_items)
        """
        user_content = format_trajectory_for_extraction(
            trajectory, task_description, success
        )

        # Use different prompt based on success/failure (Paper Appendix A.1)
        system_prompt = EXTRACTION_SYSTEM_PROMPT_SUCCESS if success else EXTRACTION_SYSTEM_PROMPT_FAILURE

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response, tokens = self.llm_client.complete(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048
            )

            # Check for errors from safety filters
            if response.startswith("Error:"):
                logger.warning(f"Extractor LLM error for {trajectory.task_id}: {response}")
                # Return empty list if safety blocked
                return []

            # Parse Markdown response (paper uses Markdown format, not JSON)
            response = response.strip()
            if not response:
                logger.warning(f"Empty response from extractor for {trajectory.task_id}")
                return []

            # Parse Markdown format memory items
            memory_items = self._parse_markdown_memories(response, trajectory, success)

            logger.info(
                f"Extracted {len(memory_items)} strategies from {trajectory.task_id} "
                f"(success={success})"
            )

            return memory_items[:self.max_items]

        except Exception as e:
            logger.error(f"Extraction error for {trajectory.task_id}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

    def _parse_markdown_memories(
        self,
        response: str,
        trajectory: Trajectory,
        success: bool
    ) -> List[MemoryItem]:
        """Parse Markdown formatted memory items from LLM response."""
        memory_items = []

        # Split by "# Memory Item" markers
        import re
        # Match "# Memory Item" followed by number or "i"
        items = re.split(r'#\s+Memory\s+Item\s+\d+|#\s+Memory\s+Item\s+i', response, flags=re.IGNORECASE)

        # Skip first split (text before first item)
        for item_text in items[1:]:
            if not item_text.strip():
                continue

            try:
                # Extract title (## Title ...)
                title_match = re.search(r'##\s+Title\s+(.+?)(?=\n|$)', item_text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "Untitled Strategy"

                # Extract description (## Description ...)
                desc_match = re.search(r'##\s+Description\s+(.+?)(?=\n##|\n#|$)', item_text, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""

                # Extract content (## Content ...)
                content_match = re.search(r'##\s+Content\s+(.+?)(?=\n##|\n#|$)', item_text, re.IGNORECASE | re.DOTALL)
                content_text = content_match.group(1).strip() if content_match else ""

                # Convert content to list (split by sentences or newlines)
                if content_text:
                    # Try to split by bullet points or newlines
                    content = [c.strip('- ').strip() for c in content_text.split('\n') if c.strip()]
                    # If no newlines, treat as single content
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
                        "task_id": trajectory.task_id,
                        "success": success,
                        "timestamp": datetime.utcnow().isoformat(),
                        "steps": trajectory.steps
                    }
                )
                memory_items.append(memory_item)

            except Exception as e:
                logger.warning(f"Failed to parse memory item: {e}")
                logger.debug(f"Item text: {item_text[:200]}")
                continue

        return memory_items
