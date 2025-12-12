"""Mind2Web agent implementation for ReasoningBank.

Agent for web navigation tasks in Mind2Web benchmark using BrowserGym.
"""
import os
import time
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from src.llm_client import LLMClient
from src.models import Action, Trajectory, MemoryItem
from src.memory import ReasoningBank


# Mind2Web agent system prompt
MIND2WEB_SYSTEM_PROMPT_BASE = """You are an expert web navigation agent. You will be given a task goal describing what you need to accomplish on a website.

Your goal is to complete the task by interacting with the web page using appropriate actions.

Available actions:
- click(element_id): Click on an element (buttons, links, dropdown options, etc.)
- type(element_id, text): Type text into an input field
- select_option(element_id, value): Select an option from a dropdown
- scroll(direction): Scroll the page (up/down)
- read_page(): Read the current page content
- navigate(url): Navigate to a URL (only when necessary)
- finish(answer): Complete the task with final answer

IMPORTANT GUIDELINES:
1. **Read the Page First**: Always read the current page to understand what elements are available
2. **Identify Target Elements**: Find the correct element IDs for buttons, links, input fields, etc.
3. **Prioritize Key Elements**: Look for search boxes, input fields, and primary action buttons FIRST
4. **Execute Actions Systematically**: Click buttons, fill forms, select options as needed
5. **Minimize Scrolling**: Only scroll if you cannot find the target element on the current view
6. **Be Efficient**: Complete the task with minimal steps - avoid unnecessary scrolling and clicking
7. **Verify Progress**: After each action, check if you're moving toward the goal

ACTION TYPES (Mind2Web operations):
- **Click**: Use for buttons, links, checkboxes, radio buttons
  Example: click(element_id="42")
- **Type**: Use for text inputs, search boxes, text areas
  Example: type(element_id="23", text="search query")
- **Select Option**: Use for dropdown menus, comboboxes
  Example: select_option(element_id="15", value="option_value")

CRITICAL FORMAT REQUIREMENT:
Your response MUST ALWAYS follow this format:

Thought: [Your reasoning about what to do next]
Action: [action_name]
Args: {"arg1": "value1", "arg2": "value2"}

OR, to complete the task:

Thought: [Final reasoning]
Action: finish
Args: {"answer": "your final answer or confirmation"}

IMPORTANT:
- ALWAYS include "Thought:" before each action
- Args MUST be complete valid JSON with both opening { and closing }
- Think step-by-step before acting
- Pay attention to element IDs and types
- If an action fails, analyze the error and try a different approach
- Be systematic and methodical in your approach

ELEMENT PRIORITIZATION:
When looking for elements to interact with:
1. **Search boxes/input fields** - Look for textbox, searchbox elements first
2. **Primary buttons** - "Search", "Submit", "Continue", "Add to Cart"
3. **Form fields** - Input fields related to your task
4. **Navigation links** - Only if you need to change pages
5. **Avoid** - Generic links, social media buttons, ads

COMMON PATTERNS:
- To search: Find [textbox] or [searchbox] → Type query → Click search [button]
- To fill forms: Find relevant [textbox] elements → Type data → Click submit [button]
- To select items: Find item [link] or [button] → Click it

ERROR HANDLING:
- If a click action times out or fails:
  * The element might be blocked by a popup/overlay
  * Look for alternative elements (e.g., different buttons with similar labels)
  * Try using read_page to get a fresh view of the page
  * Avoid excessive scrolling - scroll only if element is not visible
- If you keep failing on the same action, try a different strategy
- Don't repeat the same failed action more than twice
- Look for alternative paths to achieve the goal

EXAMPLES:

Example 1 - Searching for an item:
Task: Search for "fishing classes" in Chicago

Thought: I need to find a search box to enter my query. Let me read the page to identify search elements.
Action: read_page
Args: {}

Observation: ... [textbox] bid=123: Search events ...

Thought: I found a search textbox with bid=123. I'll type "fishing classes" into it.
Action: type
Args: {"element_id": "123", "text": "fishing classes"}

Thought: Now I need to find a location field to specify Chicago.
Action: read_page
Args: {}

Observation: ... [textbox] bid=456: Location ...

Thought: Found the location field. I'll type "Chicago".
Action: type
Args: {"element_id": "456", "text": "Chicago"}

Thought: Now I should look for a search button or submit button.
Action: read_page
Args: {}

Observation: ... [button] bid=789: Search ...

Thought: Perfect! I'll click the search button to execute the search.
Action: click
Args: {"element_id": "789"}

Example 2 - Filling a form:
Task: Add item to cart with quantity 2

Thought: I need to find the quantity input field. Let me read the page.
Action: read_page
Args: {}

Observation: ... [textbox] bid=234: Quantity ... [button] bid=567: Add to Cart ...

Thought: I see a quantity textbox (bid=234) and an Add to Cart button (bid=567). First, I'll enter the quantity.
Action: type
Args: {"element_id": "234", "text": "2"}

Thought: Now I'll click the Add to Cart button to complete the task.
Action: click
Args: {"element_id": "567"}

Example 3 - Reading the page:
Thought: I need to understand what elements are available on this page. Let me read the current page content.
Action: read_page
Args: {}

Example 2 - Clicking a button:
Thought: I can see a "Search" button with element ID 42. Let me click it to submit the search.
Action: click
Args: {"element_id": "42"}

Example 3 - Typing in a search box:
Thought: I need to search for "python tutorial". The search box has element ID 23.
Action: type
Args: {"element_id": "23", "text": "python tutorial"}

Example 4 - Selecting from dropdown:
Thought: I need to select "Category: Books" from the dropdown with element ID 15.
Action: select_option
Args: {"element_id": "15", "value": "books"}

Example 5 - Completing the task:
Thought: I have successfully completed the search and found the required item. The task is complete.
Action: finish
Args: {"answer": "Task completed successfully"}

REMEMBER:
- Work systematically through the task
- Always verify your understanding before acting
- Keep your actions focused on the task goal
- Track your progress toward completing the task
"""


class Mind2WebBrowserEnvironment:
    """
    Browser environment for Mind2Web tasks using BrowserGym.

    Similar to WebArena but adapted for Mind2Web task format.
    Mind2Web tasks involve real-world websites with:
    - Click actions on various elements
    - Type actions for text input
    - Select option actions for dropdowns
    """

    def __init__(self, task_data: Dict[str, Any], use_real_browser: bool = False):
        self.task_data = task_data
        self.use_real_browser = use_real_browser
        self.current_url = task_data.get("start_url", "")
        self.page_content = ""
        self.history = []
        self.env = None
        self.last_obs = None

        # Track actions for evaluation metrics
        self.predicted_actions = []  # Agent's predicted actions
        self.ground_truth_actions = task_data.get("actions", [])

        # Try to initialize real browser if requested
        if use_real_browser:
            try:
                self._init_browsergym()
            except Exception as e:
                logger.warning(f"Failed to initialize BrowserGym: {e}")
                logger.warning("Falling back to mock environment")
                self.use_real_browser = False

    def _init_browsergym(self):
        """Initialize BrowserGym environment for Mind2Web interaction."""
        try:
            import gymnasium as gym
            # Import browsergym for web environment
            try:
                import browsergym.core
            except ImportError:
                logger.warning("browsergym not installed, falling back to mock")
                self.use_real_browser = False
                return

            # Create a generic web browsing environment
            # Mind2Web uses real websites, so we use a generic browser
            logger.info("Creating BrowserGym environment for Mind2Web")

            # Get start URL for Mind2Web task
            # Mind2Web doesn't provide start_url, so construct from website name
            website = self.task_data.get("website", "")
            if website:
                # Most websites follow www.{name}.com pattern
                start_url = f"https://www.{website}.com"
            else:
                start_url = self.task_data.get("start_url", "https://www.google.com")

            try:
                # Create BrowserGym environment with correct parameters
                self.env = gym.make(
                    "browsergym/openended",
                    task_kwargs={"start_url": start_url},
                    viewport={"width": 1280, "height": 720},
                    timeout=90000,
                    headless=True
                )
                logger.info("✓ Created Mind2Web browser environment")
            except Exception as e:
                logger.warning(f"Could not create browsergym environment: {e}")
                self.use_real_browser = False
                return

            # Reset environment (this will navigate to start URL)
            obs, info = self.env.reset()
            self.last_obs = obs
            logger.info(f"✓ Initialized browser at: {start_url}")

            # Wait for page to fully load and render
            # Some websites use JavaScript to render content after initial load
            import time
            logger.info("Waiting for page to fully load...")
            time.sleep(5)  # Wait 5 seconds for JS to execute and page to stabilize

            # Try to wait for network idle using BrowserGym's page object
            try:
                # Access the playwright page through BrowserGym
                if hasattr(self.env.unwrapped, 'page'):
                    page = self.env.unwrapped.page
                    # Wait for network to be idle (increased timeout for slow sites)
                    page.wait_for_load_state("networkidle", timeout=30000)
                    logger.info("✓ Network idle")
            except Exception as e:
                logger.debug(f"Could not wait for network idle: {e}")

            # Perform a noop action to get fresh observation after waiting
            obs, reward, terminated, truncated, info = self.env.step("noop()")
            self.last_obs = obs
            logger.info("✓ Got fresh observation after page load")

            # Try to close common popups/modals that block interaction
            self._close_initial_popups()

        except Exception as e:
            logger.error(f"Failed to initialize BrowserGym: {e}")
            self.use_real_browser = False

    def _close_initial_popups(self):
        """
        Attempt to close common popups/modals that appear on page load.

        Common blockers:
        - Cookie consent banners
        - Chat widgets
        - Promotional popups
        - Email signup forms
        """
        if not self.last_obs:
            return

        try:
            # Extract elements that might be close buttons
            interactive_elements = self._extract_interactive_elements(self.last_obs)

            # Look for common close button patterns
            close_patterns = [
                'close', 'dismiss', 'accept', 'ok', 'got it',
                'no thanks', 'maybe later', 'continue', '×', 'x'
            ]

            for element in interactive_elements[:50]:  # Check first 50 elements
                element_lower = element.lower()

                # Check if element matches close pattern
                for pattern in close_patterns:
                    if pattern in element_lower and ('button' in element_lower or 'link' in element_lower):
                        # Extract bid
                        if 'bid=' in element:
                            try:
                                bid = element.split('bid=')[1].split(':')[0].split(']')[0].strip()
                                logger.info(f"Attempting to close popup: {element[:80]}")

                                # Try to click it
                                try:
                                    obs, _, _, _, _ = self.env.step(f"click('{bid}')")
                                    self.last_obs = obs
                                    logger.info(f"✓ Closed popup/modal")
                                    return  # Only close one popup at a time
                                except Exception as e:
                                    logger.debug(f"Could not click popup close button: {e}")
                                    continue
                            except:
                                continue
                        break
        except Exception as e:
            logger.debug(f"Error while closing popups: {e}")

    def _prioritize_elements(self, elements: List[str]) -> List[str]:
        """
        Prioritize interactive elements to show most relevant ones first.

        Priority order:
        1. Search boxes and text inputs
        2. Primary action buttons (Search, Submit, Add to Cart, etc.)
        3. Form elements (textbox, combobox, etc.)
        4. Navigation buttons and links
        5. Other interactive elements
        """
        high_priority = []
        medium_priority = []
        low_priority = []

        # Keywords for high priority elements
        high_keywords = [
            'search', 'textbox', 'searchbox', 'input', 'query',
            'submit', 'continue', 'next', 'add to cart', 'checkout',
            'sign in', 'login', 'register', 'buy', 'purchase'
        ]

        # Keywords for medium priority
        medium_keywords = [
            'button', 'combobox', 'select', 'option', 'dropdown',
            'form', 'menu', 'filter', 'sort'
        ]

        for element in elements:
            element_lower = element.lower()

            # Check if high priority
            if any(keyword in element_lower for keyword in high_keywords):
                high_priority.append(element)
            # Check if medium priority
            elif any(keyword in element_lower for keyword in medium_keywords):
                medium_priority.append(element)
            # Everything else is low priority
            else:
                low_priority.append(element)

        # Return prioritized list
        return high_priority + medium_priority + low_priority

    def _extract_interactive_elements(self, obs: Dict[str, Any]) -> List[str]:
        """
        Extract interactive elements from BrowserGym observation.

        Uses both accessibility tree (axtree_object) and extra_element_properties
        to find all clickable/interactable elements on the page.

        Args:
            obs: BrowserGym observation dictionary

        Returns:
            List of formatted element strings with bid, role, and text
        """
        interactive_elements = []

        try:
            # Get extra_element_properties - contains bid -> {clickable, visibility, bbox}
            extra_props = obs.get("extra_element_properties", {})

            # Get accessibility tree - contains semantic information
            axtree = obs.get("axtree_object", {})

            # Build a mapping of bid -> element info from axtree
            bid_to_element = {}

            if isinstance(axtree, str):
                # Parse JSON string if needed
                import json
                try:
                    axtree = json.loads(axtree)
                except:
                    pass

            if isinstance(axtree, dict):
                # BrowserGym axtree format: list of nodes with role, name, description, etc.
                nodes = axtree.get("nodes", [])

                if isinstance(nodes, str):
                    try:
                        import json
                        nodes = json.loads(nodes)
                    except:
                        nodes = []

                for node in nodes:
                    if not isinstance(node, dict):
                        continue

                    # BrowserGym uses "browsergym_id" as the main element identifier
                    bid = node.get("browsergym_id", None)

                    # Fallback to other possible ID fields
                    if not bid:
                        bid = node.get("backend_node_id", None)
                    if not bid:
                        bid = node.get("nodeId", None)

                    if bid:
                        # Extract role
                        role = node.get("role", {})
                        if isinstance(role, dict):
                            role_value = role.get("value", "")
                        else:
                            role_value = str(role) if role else ""

                        # Extract name (visible text)
                        name = node.get("name", {})
                        if isinstance(name, dict):
                            name_value = name.get("value", "")
                        else:
                            name_value = str(name) if name else ""

                        # Store element info
                        bid_to_element[str(bid)] = {
                            "role": role_value,
                            "name": name_value,
                            "node": node
                        }

            # Extract elements from axtree with interactive semantic roles
            # This is the primary source as extra_element_properties may be incomplete
            interactive_roles = {
                'button', 'link', 'textbox', 'searchbox', 'combobox',
                'search', 'menuitem', 'tab', 'checkbox', 'radio', 'input',
                'option', 'listitem'  # Added for dropdown options and list items
            }

            for bid, elem_info in bid_to_element.items():
                role = elem_info.get("role", "").lower()
                if role in interactive_roles:
                    name = elem_info.get("name", "")
                    # Clean up name
                    if name:
                        name = name.strip()[:100]
                        interactive_elements.append(f"[{role}] bid={bid}: {name}")
                    else:
                        interactive_elements.append(f"[{role}] bid={bid}")

            # Also check extra_element_properties for clickable elements
            # (in case some don't have proper semantic roles)
            if isinstance(extra_props, dict):
                existing_bids = {elem.split("bid=")[1].split(":")[0].split("]")[0]
                                for elem in interactive_elements if "bid=" in elem}

                for bid, props in extra_props.items():
                    if not isinstance(props, dict):
                        continue

                    # Include if clickable, regardless of visibility
                    # (websites may have anti-bot detection affecting visibility)
                    if props.get("clickable", False) and str(bid) not in existing_bids:
                        elem_info = bid_to_element.get(str(bid), {})
                        role = elem_info.get("role", "element")
                        name = elem_info.get("name", "")

                        if name:
                            name = name.strip()[:100]
                            interactive_elements.append(f"[{role}] bid={bid}: {name}")
                        else:
                            interactive_elements.append(f"[{role}] bid={bid}")

            logger.debug(f"Extracted {len(interactive_elements)} interactive elements")
            return interactive_elements

        except Exception as e:
            logger.error(f"Error extracting interactive elements: {e}", exc_info=True)
            return []

    def execute_action(self, action_type: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Execute an action in the browser environment.

        Args:
            action_type: Type of action (click, type, select_option, etc.)
            args: Action arguments

        Returns:
            Tuple of (observation, success)
        """
        # Record predicted action for evaluation
        self.predicted_actions.append({
            "action_type": action_type,
            "args": args
        })

        if self.use_real_browser and self.env:
            return self._execute_real_action(action_type, args)
        else:
            return self._execute_mock_action(action_type, args)

    def _execute_real_action(self, action_type: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute action in real BrowserGym environment."""
        try:
            # Convert Mind2Web actions to BrowserGym format
            # BrowserGym uses the bid directly (no prefix needed)
            if action_type == "click":
                element_id = args.get("element_id", "")
                # BrowserGym expects just the bid number
                action_str = f"click('{element_id}')"
            elif action_type == "type":
                element_id = args.get("element_id", "")
                text = args.get("text", "")
                # Escape single quotes in text
                text = text.replace("'", "\\'")
                action_str = f"fill('{element_id}', '{text}')"
            elif action_type == "select_option":
                element_id = args.get("element_id", "")
                value = args.get("value", "")
                action_str = f"select_option('{element_id}', '{value}')"
            elif action_type == "scroll":
                direction = args.get("direction", "down")
                delta_y = 500 if direction == "down" else -500
                action_str = f"scroll(0, {delta_y})"
            elif action_type == "navigate":
                url = args.get("url", "")
                action_str = f"goto('{url}')"
            elif action_type == "read_page":
                action_str = "noop()"  # Read from observation
            else:
                return f"Unknown action type: {action_type}", False

            logger.debug(f"Executing BrowserGym action: {action_str}")

            # Execute action with timeout handling
            try:
                obs, reward, terminated, truncated, info = self.env.step(action_str)
                self.last_obs = obs
            except Exception as e:
                # If action times out or fails, try to recover
                if "TimeoutError" in str(e) or "Timeout" in str(e):
                    logger.warning(f"Action timed out: {action_str}")
                    # Return current observation and mark as failed
                    return f"Action timed out after 60s. Element may be blocked by overlay or not interactable.", False
                else:
                    raise

            # Extract observation text from BrowserGym format
            if isinstance(obs, dict):
                # Check for errors first
                error = obs.get("last_action_error", "")
                if error:
                    return f"Action failed: {error}", False

                # Get current URL
                url = obs.get("url", "")
                page_info = f"Current URL: {url}\n\n"

                # Extract interactive elements using BrowserGym's observation format
                interactive_elements = self._extract_interactive_elements(obs)

                if interactive_elements:
                    # Prioritize important elements
                    prioritized = self._prioritize_elements(interactive_elements)

                    page_info += "Interactive elements:\n"
                    page_info += "\n".join(prioritized[:80])  # Show prioritized elements
                    page_info += "\n"
                    if len(prioritized) > 80:
                        page_info += f"\n... and {len(prioritized) - 80} more elements\n"
                else:
                    page_info += "[Page loaded - no interactive elements detected]\n"
                    page_info += "Try scrolling to load more content or wait for page to fully render.\n"

                return page_info[:3000], True  # Limit total observation size
            else:
                return str(obs)[:3000], True

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return f"Error: {str(e)}", False

    def _execute_mock_action(self, action_type: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Execute action in mock environment (for testing without real browser)."""
        # Mock execution for development/testing
        self.history.append((action_type, args))

        if action_type == "read_page":
            mock_page = f"""
[Mock Page Content]
URL: {self.current_url}
Task: {self.task_data.get('confirmed_task', 'Unknown')}

Available elements:
- Button "Search" (id: btn_search)
- Textbox "Query" (id: input_query)
- Link "Home" (id: link_home)

This is a mock environment. Actions are recorded but not executed.
            """
            return mock_page, True
        elif action_type == "click":
            return f"Clicked element {args.get('element_id', 'unknown')}", True
        elif action_type == "type":
            return f"Typed '{args.get('text', '')}' into element {args.get('element_id', 'unknown')}", True
        elif action_type == "select_option":
            return f"Selected option '{args.get('value', '')}' in element {args.get('element_id', 'unknown')}", True
        elif action_type == "navigate":
            url = args.get("url", "")
            self.current_url = url
            return f"Navigated to {url}", True
        elif action_type == "scroll":
            return f"Scrolled {args.get('direction', 'down')}", True
        else:
            return f"Unknown action: {action_type}", False

    def get_predicted_actions(self) -> List[Dict[str, Any]]:
        """Get list of predicted actions for evaluation."""
        return self.predicted_actions

    def get_ground_truth_actions(self) -> List[Dict[str, Any]]:
        """Get ground truth actions from task data."""
        return self.ground_truth_actions

    def reset(self):
        """Reset environment state."""
        self.predicted_actions = []
        self.history = []
        if self.use_real_browser and self.env:
            obs, info = self.env.reset()
            self.last_obs = obs


class Mind2WebAgent:
    """
    Mind2Web agent using ReAct reasoning with optional ReasoningBank memory.

    Performs web navigation tasks by:
    1. Reading page content
    2. Identifying target elements
    3. Executing actions (click, type, select)
    4. Tracking progress toward goal
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_bank: Optional[ReasoningBank] = None,
        max_steps: int = 15,
        timeout: int = 300,
        use_real_browser: bool = False
    ):
        """
        Initialize Mind2Web agent.

        Args:
            llm_client: LLM client for generating actions
            memory_bank: Optional ReasoningBank for memory augmentation
            max_steps: Maximum steps per task
            timeout: Task timeout in seconds
            use_real_browser: Use real BrowserGym environment
        """
        self.llm_client = llm_client
        self.memory_bank = memory_bank
        self.max_steps = max_steps
        self.timeout = timeout
        self.use_real_browser = use_real_browser

    def run_task(
        self,
        task_data: Dict[str, Any],
        seed: int = 42
    ) -> Trajectory:
        """
        Run Mind2Web task.

        Args:
            task_data: Mind2Web task dictionary
            seed: Random seed

        Returns:
            Trajectory with actions and results
        """
        logger.info(f"Starting Mind2Web task: {task_data.get('annotation_id', 'unknown')}")
        logger.info(f"Task: {task_data.get('confirmed_task', 'unknown')[:100]}...")

        # Create browser environment
        env = Mind2WebBrowserEnvironment(task_data, self.use_real_browser)

        # Build system prompt
        system_prompt = MIND2WEB_SYSTEM_PROMPT_BASE

        # Get task description
        task_description = task_data.get("confirmed_task", "")

        # Retrieve relevant memories if using ReasoningBank
        retrieved_memories = []
        if self.memory_bank:
            retrieved_memories = self.memory_bank.retrieve(
                query=task_description,
                k=5
            )
            if retrieved_memories:
                memory_context = "\n\n=== RELEVANT EXPERIENCE FROM MEMORY BANK ===\n"
                for i, mem in enumerate(retrieved_memories, 1):
                    memory_context += f"\nMemory {i}:\n{mem.content}\n"
                memory_context += "\n=== END OF MEMORY BANK ===\n"
                system_prompt = system_prompt + memory_context

        # Initialize trajectory
        trajectory = Trajectory(
            task_id=task_data.get("annotation_id", ""),
            task_description=task_description,
            actions=[],
            observations=[],
            steps=0,
            success=False,
            final_answer="",
            retrieved_memories=[m.model_dump() for m in retrieved_memories],
            tokens={"input": 0, "output": 0},
            walltime=0.0
        )

        start_time = time.time()
        conversation_history = []

        # Add system prompt
        conversation_history.append({
            "role": "system",
            "content": system_prompt
        })

        # Add task instruction
        initial_message = f"Task Goal: {task_description}\n\nBegin by reading the page to understand what elements are available."
        conversation_history.append({
            "role": "user",
            "content": initial_message
        })

        # ReAct loop
        for step in range(self.max_steps):
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"Task timed out after {self.timeout}s")
                break

            logger.info(f"Step {step + 1}/{self.max_steps}")

            # Get agent response
            try:
                response = self.llm_client.generate(
                    messages=conversation_history,
                    temperature=0.0,
                    max_tokens=500
                )

                # Track tokens
                trajectory.tokens["input"] += response.get("usage", {}).get("prompt_tokens", 0)
                trajectory.tokens["output"] += response.get("usage", {}).get("completion_tokens", 0)

                agent_response = response["content"]
                logger.info(f"Agent: {agent_response[:200]}...")

                # Parse action
                action = self._parse_action(agent_response)

                if not action:
                    logger.warning("Failed to parse action, retrying...")
                    conversation_history.append({
                        "role": "assistant",
                        "content": agent_response
                    })
                    conversation_history.append({
                        "role": "user",
                        "content": "ERROR: Invalid action format. Please use the correct format:\nThought: ...\nAction: ...\nArgs: {...}"
                    })
                    continue

                # Store action
                trajectory.actions.append(action)
                trajectory.steps += 1

                # Check if finish action
                if action.tool == "finish":
                    trajectory.final_answer = action.args.get("answer", "")
                    trajectory.success = True  # Will be evaluated by evaluator
                    logger.info(f"Task finished: {trajectory.final_answer}")
                    break

                # Execute action
                observation, success = env.execute_action(action.tool, action.args)
                trajectory.observations.append(observation)

                logger.info(f"Observation: {observation[:200]}...")

                # Update conversation
                conversation_history.append({
                    "role": "assistant",
                    "content": agent_response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation[:2000]}"  # Limit observation size
                })

            except Exception as e:
                logger.error(f"Error in step {step + 1}: {e}")
                trajectory.observations.append(f"Error: {str(e)}")
                break

        trajectory.walltime = time.time() - start_time

        # Store predicted actions for evaluation
        trajectory.predicted_actions = env.get_predicted_actions()
        trajectory.ground_truth_actions = env.get_ground_truth_actions()

        logger.info(f"Task completed in {trajectory.steps} steps, {trajectory.walltime:.1f}s")

        return trajectory

    def _parse_action(self, response: str) -> Optional[Action]:
        """Parse action from agent response."""
        try:
            # Find Thought, Action, Args
            lines = response.strip().split('\n')

            thought = ""
            action_type = ""
            args = {}

            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("Thought:"):
                    thought = line.replace("Thought:", "").strip()
                elif line.startswith("Action:"):
                    action_type = line.replace("Action:", "").strip()
                elif line.startswith("Args:"):
                    # Parse JSON args
                    args_str = line.replace("Args:", "").strip()
                    # Handle potential multi-line JSON
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith(("Thought:", "Action:", "Args:")):
                        args_str += " " + lines[i + 1].strip()

                    import json
                    args = json.loads(args_str)

            if not action_type:
                return None

            return Action(
                tool=action_type,
                args=args
            )

        except Exception as e:
            logger.warning(f"Failed to parse action: {e}")
            return None
