"""SWE-bench agent with Bash-only environment for ReasoningBank.

Implements mini-SWE-Agent style ReAct agent for repository-level issue resolution.
"""
import os
import subprocess
import time
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from src.llm_client import LLMClient
from src.models import Action, Trajectory, MemoryItem
from src.memory import ReasoningBank


# System prompt for SWE-bench agent
SWEBENCH_SYSTEM_PROMPT_BASE = """You are an expert software engineer working on fixing bugs and resolving issues in open-source repositories.

Your goal is to analyze the issue description, explore the repository, identify the root cause, and implement a fix.

You interact with the repository using bash commands. You have full access to the repository files and can run any bash command.

Available actions:
- Execute any bash command (e.g., ls, cd, cat, grep, find, git, python, pytest)
- Read files, search for patterns, run tests
- Make code changes by editing files
- Verify your changes by running tests

IMPORTANT GUIDELINES:
1. **Understand the Issue**: Carefully read the problem statement and hints
2. **Explore the Codebase**: Use ls, find, grep to locate relevant files
3. **Identify Root Cause**: Read the code and understand what's causing the issue
4. **Implement Fix**: Make minimal, targeted changes to fix the issue
5. **Generate Patch**: After making changes, use `git diff` and submit with finish action
6. **Be Efficient**: Submit your patch as soon as the fix is made and verified (via grep/cat)

CRITICAL: Testing is OPTIONAL. If tests are complex or require building, SKIP testing and submit the patch immediately after verifying the code change was made correctly.

CRITICAL FORMAT REQUIREMENT:
Your response MUST ALWAYS follow this format:

Thought: [Your reasoning about what to do next]
Action: bash
Args: {"command": "your bash command here"}

OR, to submit your final patch:

Thought: [Explanation of your fix]
Action: finish
Args: {"patch": "your git diff output"}

IMPORTANT RULES:
- Always include "Thought:" before each action
- Use proper bash syntax in your commands
- Don't use interactive commands (like vim, nano) - use Python scripts, sed, or perl for file editing
- Check command output carefully before proceeding
- If a command fails, analyze the error and try a different approach
- After making code changes, IMMEDIATELY verify with grep or cat, then use `git diff` and submit
- DO NOT spend time testing if it requires complex setup (building, installing dependencies)
- If approaching step 25+, SUBMIT YOUR PATCH NOW - don't try to test

EXAMPLES:

Example 1 - Exploring the repository:
Thought: I need to understand the repository structure first. Let me list the top-level directories.
Action: bash
Args: {"command": "ls -la"}

Example 2 - Finding relevant files:
Thought: The issue mentions a bug in the parser. Let me search for parser-related files.
Action: bash
Args: {"command": "find . -name '*parser*' -type f"}

Example 3 - Reading a file:
Thought: Let me examine the parser.py file to understand the implementation.
Action: bash
Args: {"command": "cat src/parser.py"}

Example 4 - Making a change:
Thought: I need to fix the off-by-one error in line 42. Let me use sed to update it.
Action: bash
Args: {"command": "sed -i 's/range(len(items))/range(len(items) - 1)/g' src/parser.py"}

Example 5 - Verifying the change:
Thought: Let me verify the change was applied correctly.
Action: bash
Args: {"command": "grep -n 'range(len(items) - 1)' src/parser.py"}

Example 6 - Generating and submitting the patch:
Thought: The change is verified. Let me generate the git diff without index line and submit immediately.
Action: bash
Args: {"command": "git diff HEAD"}

[After seeing the diff output]
Thought: The patch looks correct. Submitting the final fix NOW. Note: I will NOT include the 'index' line in my patch.
Action: finish
Args: {"patch": "diff --git a/src/parser.py b/src/parser.py\\n--- a/src/parser.py\\n+++ b/src/parser.py\\n@@ -39,7 +39,7 @@ def parse(items):\\n-    for i in range(len(items)):\\n+    for i in range(len(items) - 1):\\n         process(items[i])"}

REMEMBER:
- Work systematically and methodically
- Always verify your understanding before making changes
- Keep your fix minimal and focused on the issue
- Submit patches QUICKLY - verify the code change with grep, then git diff and finish
- DO NOT waste time on testing if it's complex - just submit the patch
- Your PRIMARY goal is to generate a git diff patch, not to run tests
- IMPORTANT: When submitting patches, do NOT include the 'index' line (e.g., "index abc123..def456")
- Use `git diff HEAD` to generate clean patches, or manually remove the index line from git diff output
"""


class BashEnvironment:
    """Simple bash environment for SWE-bench tasks."""

    def __init__(self, repo_path: str):
        """
        Initialize bash environment.

        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        self.current_dir = repo_path

    def execute(self, command: str, timeout: int = 60) -> Tuple[str, int]:
        """
        Execute a bash command.

        Args:
            command: Bash command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (output, return_code)
        """
        try:
            # Execute command in the repository directory
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.current_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return output, result.returncode

        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds", -1
        except Exception as e:
            return f"Error executing command: {str(e)}", -1

    def reset(self):
        """Reset environment state."""
        self.current_dir = self.repo_path


class SWEBenchAgent:
    """SWE-bench agent with bash-only environment."""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_bank: Optional[ReasoningBank] = None,
        max_steps: int = 30,
        timeout: int = 900
    ):
        """
        Initialize SWE-bench agent.

        Args:
            llm_client: LLM client for agent reasoning
            memory_bank: Optional ReasoningBank for memory-augmented agent
            max_steps: Maximum number of steps per task
            timeout: Task timeout in seconds
        """
        self.llm_client = llm_client
        self.memory_bank = memory_bank
        self.max_steps = max_steps
        self.timeout = timeout

    def _build_system_prompt(
        self,
        retrieved_memories: Optional[List[MemoryItem]] = None
    ) -> str:
        """
        Build system prompt with optional memory augmentation.

        Args:
            retrieved_memories: Retrieved memory items from ReasoningBank

        Returns:
            Complete system prompt
        """
        prompt = SWEBENCH_SYSTEM_PROMPT_BASE

        if retrieved_memories and len(retrieved_memories) > 0:
            prompt += "\n\n=== REASONING BANK: PAST EXPERIENCES ===\n\n"
            prompt += "Below are relevant strategies from past tasks that may be helpful.\n"
            prompt += "These are actionable patterns proven effective in similar scenarios.\n\n"
            prompt += "HOW TO USE:\n"
            prompt += "1. Review if applicable to current task\n"
            prompt += "2. State which memory you're applying\n"
            prompt += "3. Follow or adapt approach\n\n"

            for i, mem in enumerate(retrieved_memories, 1):
                prompt += f"MEMORY {i}: {mem.title}\n"
                prompt += f"  Summary: {mem.description}\n"
                prompt += f"  Strategy:\n"
                for step in mem.content:
                    prompt += f"    - {step}\n"

                # Add provenance indicator
                if mem.provenance.get("success"):
                    prompt += "  ✓ Validated from successful experience\n"
                else:
                    prompt += "  ⚠ Learned from failure - preventative strategy\n"
                prompt += "\n"

        return prompt

    def run_task(
        self,
        task_description: str,
        repo_path: str,
        base_commit: str,
        seed: int = 42
    ) -> Trajectory:
        """
        Run agent on a SWE-bench task.

        Args:
            task_description: Task description (problem statement)
            repo_path: Path to repository
            base_commit: Base commit hash
            seed: Random seed

        Returns:
            Trajectory of agent execution
        """
        start_time = time.time()

        # Initialize trajectory
        trajectory = Trajectory(
            task_id=f"swebench_{base_commit[:8]}",
            task_description=task_description,
            seed=seed,
            backbone=getattr(self.llm_client, 'model', 'unknown')
        )

        # Initialize bash environment
        env = BashEnvironment(repo_path)

        # Retrieve memories if using ReasoningBank
        retrieved_memories = []
        if self.memory_bank:
            try:
                retrieved_memories = self.memory_bank.retrieve(
                    query=task_description,
                    k=3  # Retrieve top-3 most relevant memories
                )
                trajectory.retrieved_memories = retrieved_memories
                logger.info(f"Retrieved {len(retrieved_memories)} memories from ReasoningBank")
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")

        # Build system prompt
        system_prompt = self._build_system_prompt(retrieved_memories)

        # Agent loop
        conversation_history = []
        current_observation = f"Task: {task_description}\n\nRepository initialized at: {repo_path}\nBase commit: {base_commit}\n\nYou can now start working on the issue."
        final_patch = None

        # Track command history for loop detection
        command_history = []

        for step in range(self.max_steps):
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"Task timeout after {self.timeout}s")
                break

            # Build prompt with conversation history
            user_prompt = f"Observation: {current_observation}\n\n"

            # Add urgency reminder when approaching max steps
            if step >= self.max_steps - 5:
                user_prompt += f"⚠️ WARNING: You are at step {step}/{self.max_steps}. If you have made code changes, SUBMIT YOUR PATCH NOW using 'git diff' and 'finish' action. DO NOT test.\n\n"

            user_prompt += "What's your next action?"

            # Get agent response
            try:
                # Build messages for LLM client
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(conversation_history)
                messages.append({"role": "user", "content": user_prompt})

                response, tokens = self.llm_client.complete(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048
                )

                # Track token usage
                trajectory.tokens["input"] += tokens.get("input", 0)
                trajectory.tokens["output"] += tokens.get("output", 0)

                # Parse response
                thought, action_name, args = self._parse_response(response)

                # Store in trajectory
                trajectory.thoughts.append(thought)
                action = Action(tool=action_name, args=args)
                trajectory.actions.append(action)
                trajectory.observations.append(current_observation)

                # Update conversation history
                conversation_history.append({"role": "user", "content": user_prompt})
                conversation_history.append({"role": "assistant", "content": response})

                # Execute action
                if action_name == "finish":
                    final_patch = args.get("patch", "")
                    trajectory.final_answer = final_patch
                    trajectory.success = True  # Will be validated later
                    trajectory.steps = step + 1
                    logger.info(f"Agent finished after {step + 1} steps")
                    break

                elif action_name == "bash":
                    command = args.get("command", "")

                    # Track command for loop detection
                    command_history.append(command)

                    output, return_code = env.execute(command)

                    # Prepare next observation
                    current_observation = f"Command: {command}\n"
                    current_observation += f"Exit code: {return_code}\n"
                    current_observation += f"Output:\n{output}"

                    # LOOP DETECTION: Check if agent is repeating similar commands
                    if step >= 8 and len(command_history) >= 6:
                        is_looping, loop_msg = self._detect_command_loop(command_history)
                        if is_looping:
                            logger.warning(f"🔄 Command loop detected at step {step}: {loop_msg}")

                            recovery_hint = f"""

⚠️⚠️⚠️ LOOP DETECTED ⚠️⚠️⚠️

You are repeating similar commands: {loop_msg}

STOP and try a DIFFERENT approach:
1. If sed/Python edits keep failing → Try a simpler fix or use 'git diff' to see current changes
2. If repeatedly viewing the same file → Make your edit NOW and submit
3. If you've made code changes → Run 'git diff' and submit with 'finish' action
4. If approaching step 20+ → SUBMIT YOUR PATCH NOW, don't perfect it

Your current approach is NOT working. Try something completely different or submit what you have!
"""
                            current_observation += recovery_hint

                            # If loop persists, force submission
                            if step >= 20:
                                logger.error(f"❌ Persistent loop at step {step}, forcing submission")
                                current_observation += "\n\n⚠️ CRITICAL: Step 20+ reached with looping detected. You MUST submit your patch NOW using 'git diff' and 'finish' action."

                else:
                    current_observation = f"Error: Unknown action '{action_name}'"

            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                current_observation = f"Error: {str(e)}"
                trajectory.steps = step + 1
                break

        # Finalize trajectory
        trajectory.steps = min(step + 1, self.max_steps)
        trajectory.walltime = time.time() - start_time

        if not trajectory.final_answer and step >= self.max_steps - 1:
            logger.warning(f"Task reached max steps ({self.max_steps})")
            trajectory.success = False

        return trajectory

    def _detect_command_loop(self, command_history: List[str], window: int = 6) -> Tuple[bool, str]:
        """
        Detect if agent is stuck repeating similar bash commands.

        Args:
            command_history: List of executed commands
            window: Number of recent commands to check

        Returns:
            Tuple of (is_looping, reason)
        """
        if len(command_history) < window:
            return False, ""

        recent_commands = command_history[-window:]

        # Normalize commands for comparison (remove minor variations)
        def normalize_command(cmd: str) -> str:
            """Simplify command for pattern matching."""
            # Remove leading/trailing whitespace
            cmd = cmd.strip()
            # Extract the main command (first word)
            main_cmd = cmd.split()[0] if cmd.split() else cmd
            # For cat/grep/sed, include the target file
            if main_cmd in ['cat', 'grep', 'sed', 'python']:
                # Try to extract filename pattern
                import re
                file_match = re.search(r'[\w/]+\.py', cmd)
                if file_match:
                    return f"{main_cmd}:{file_match.group()}"
            return main_cmd

        normalized = [normalize_command(cmd) for cmd in recent_commands]

        # Check for repeated command patterns
        unique_commands = len(set(normalized))

        if unique_commands <= 2:
            # Very low command diversity
            most_common = max(set(normalized), key=normalized.count)
            count = normalized.count(most_common)

            if count >= window - 1:
                return True, f"Repeated command pattern '{most_common}' {count} times"

        # Check for alternating patterns (e.g., cat file, sed file, cat file, sed file)
        if unique_commands == 2:
            cmd_types = list(set(normalized))
            # Check if alternating
            alternating = all(
                normalized[i] != normalized[i+1]
                for i in range(len(normalized)-1)
            )
            if alternating:
                return True, f"Alternating between '{cmd_types[0]}' and '{cmd_types[1]}'"

        return False, ""

    def _parse_response(self, response: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Parse agent response to extract thought, action, and args.

        Args:
            response: Raw agent response

        Returns:
            Tuple of (thought, action_name, args)
        """
        import json
        import re

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_name = action_match.group(1).strip() if action_match else "unknown"

        # Extract args - IMPROVED ROBUST PARSING
        args = {}

        # Try to find Args: line and extract JSON
        args_start = response.find("Args:")
        if args_start != -1:
            # Find the JSON object starting with {
            json_start = response.find("{", args_start)
            if json_start != -1:
                # Use a more robust approach: find matching closing brace
                brace_count = 0
                json_end = json_start
                in_string = False
                escape_next = False

                for i in range(json_start, len(response)):
                    char = response[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string

                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                json_str = response[json_start:json_end]

                # Try to parse JSON
                try:
                    args = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse args as JSON: {e}")

                    # Fallback: try to extract patch/command manually
                    # This handles cases where LLM generates malformed JSON
                    if action_name == "finish":
                        # Try to extract patch field
                        patch_match = re.search(r'"patch"\s*:\s*"(.+?)"(?=\s*[,}])', json_str, re.DOTALL)
                        if patch_match:
                            patch_content = patch_match.group(1)
                            # Unescape the patch
                            patch_content = patch_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            args = {"patch": patch_content}
                            logger.info("Successfully extracted patch using fallback parser")
                    elif action_name == "bash":
                        # Try to extract command field
                        cmd_match = re.search(r'"command"\s*:\s*"(.+?)"(?=\s*[,}])', json_str, re.DOTALL)
                        if cmd_match:
                            command = cmd_match.group(1)
                            # Unescape the command
                            command = command.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            args = {"command": command}
                            logger.info("Successfully extracted command using fallback parser")

        return thought, action_name, args
