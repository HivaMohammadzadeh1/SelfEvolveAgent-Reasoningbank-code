"""
WebArena dataset loader.

Loads actual WebArena benchmark tasks from config files.
Based on: https://github.com/web-arena-x/webarena
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class WebArenaDataset:
    """
    Load WebArena benchmark tasks from config files.
    
    WebArena consists of 812 tasks across 6 domains:
    - shopping (e-commerce): 187 tasks
    - admin (CMS): 182 tasks  
    - gitlab: 180 tasks
    - reddit: 106 tasks
    - map: 128 tasks
    - multi (cross-domain): 29 tasks
    
    Each task is defined in a JSON config file with:
    - task_id: unique identifier
    - intent: natural language task description
    - require_login: whether login is needed
    - storage_state: login credentials path
    - start_url: initial URL
    - geolocation: coordinates if needed
    - intent_template: template for task description
    - instantiation_dict: variables for template
    - intent_template_id: template identifier
    """
    
    # Task counts per subset (from WebArena paper)
    SUBSET_COUNTS = {
        "shopping": 187,
        "admin": 182,
        "gitlab": 180,
        "reddit": 106,
        "map": 128,
        "multi": 29
    }
    
    def __init__(self, data_dir: str = "data/webarena"):
        """
        Initialize WebArena dataset loader.
        
        Args:
            data_dir: Directory containing WebArena config files
        """
        self.data_dir = Path(data_dir)
        self.config_dir = self.data_dir / "config_files"
        self.tasks = []
        
    def load_tasks(
        self,
        subsets: Optional[List[str]] = None,
        task_ids: Optional[List[int]] = None,
        max_multi_tasks: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load WebArena tasks from config files.
        
        Args:
            subsets: List of subsets to load (e.g., ['shopping', 'admin'])
                    If None, loads all subsets except 'map' (as per PRD)
            task_ids: Specific task IDs to load. If None, loads all tasks.
            max_multi_tasks: Maximum number of multi-domain tasks to load.
                           Paper uses 29, but WebArena has 48 total.
        
        Returns:
            List of task dictionaries
        """
        # Clear previous tasks
        self.tasks = []
        
        if subsets is None:
            # Default: all subsets except map (as per PRD)
            subsets = ["shopping", "admin", "gitlab", "reddit", "multi"]
        
        # Check if config directory exists
        if not self.config_dir.exists():
            logger.error(f"WebArena config directory not found: {self.config_dir}")
            logger.info("Please download WebArena data:")
            logger.info("  git clone https://github.com/web-arena-x/webarena.git data/webarena_repo")
            logger.info("  python data/webarena_repo/scripts/generate_test_data.py")
            logger.info("  cp -r data/webarena_repo/config_files data/webarena/")
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        # Load tasks from config files
        config_files = sorted(self.config_dir.glob("*.json"))
        
        if not config_files:
            logger.error(f"No config files found in {self.config_dir}")
            raise FileNotFoundError(f"No config files in {self.config_dir}")
        
        logger.info(f"Found {len(config_files)} config files")
        
        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                
                # Extract task ID from filename (e.g., "123.json" -> 123)
                task_id = int(config_file.stem)
                
                # Filter by task_ids if specified
                if task_ids is not None and task_id not in task_ids:
                    continue
                
                # Determine subset from start_url or other indicators
                subset = self._get_task_subset(config)
                
                # Filter by subset
                if subset not in subsets:
                    continue
                
                # Create task dictionary with None-safe handling
                description = config.get("intent", "") or ""
                start_url = config.get("start_url", "") or ""
                eval_spec = config.get("eval") or {}

                # Skip tasks with missing critical fields
                if not description and not config.get("intent_template"):
                    logger.warning(f"Skipping task {task_id}: no description or intent_template")
                    continue

                # Extract reference answer from eval section
                reference_answer = None
                if eval_spec:
                    # Try reference_answer_raw_annotation first
                    if "reference_answer_raw_annotation" in eval_spec:
                        reference_answer = str(eval_spec["reference_answer_raw_annotation"])
                    # Try reference_answers dict
                    elif "reference_answers" in eval_spec and eval_spec["reference_answers"]:
                        ref_answers = eval_spec["reference_answers"]
                        if isinstance(ref_answers, dict):
                            # Try different fields in order of preference
                            for field in ["exact_match", "must_include", "fuzzy_match", "string_match"]:
                                if field in ref_answers and ref_answers[field]:
                                    value = ref_answers[field]
                                    # Handle list values (like must_include)
                                    if isinstance(value, list) and value:
                                        reference_answer = ", ".join(str(v) for v in value)
                                    else:
                                        reference_answer = str(value)
                                    break

                    # If still no reference, try extracting from program_html
                    if not reference_answer and "program_html" in eval_spec:
                        program_html = eval_spec["program_html"]
                        if isinstance(program_html, list) and program_html:
                            # Check first program_html entry for required_contents
                            first_check = program_html[0]
                            if isinstance(first_check, dict):
                                required = first_check.get("required_contents", {})
                                if isinstance(required, dict):
                                    # Try different match types in order of preference
                                    for field in ["exact_match", "must_include", "fuzzy_match", "string_match"]:
                                        if field in required and required[field]:
                                            value = required[field]
                                            # Handle list values (like must_include)
                                            if isinstance(value, list) and value:
                                                reference_answer = ", ".join(str(v) for v in value)
                                            else:
                                                reference_answer = str(value)
                                            break

                                    # If still no reference, try must_exclude (negative reference)
                                    if not reference_answer and "must_exclude" in required and required["must_exclude"]:
                                        must_exclude = required["must_exclude"]
                                        if isinstance(must_exclude, list) and must_exclude:
                                            reference_answer = "NOT: " + ", ".join(str(v) for v in must_exclude)
                                        else:
                                            reference_answer = "NOT: " + str(must_exclude)

                task = {
                    "task_id": f"{subset}_{task_id:03d}",
                    "numeric_id": task_id,
                    "subset": subset,
                    "description": description,
                    "start_url": start_url,
                    "require_login": config.get("require_login", False) or False,
                    "storage_state": config.get("storage_state"),
                    "geolocation": config.get("geolocation"),
                    "intent_template": config.get("intent_template"),
                    "intent_template_id": config.get("intent_template_id"),
                    "instantiation_dict": config.get("instantiation_dict") or {},
                    "eval": eval_spec,
                    "config_file": str(config_file),
                    "reference_answer": reference_answer,
                    "ground_truth": reference_answer,  # Keep for backward compatibility
                }
                
                self.tasks.append(task)
                
            except Exception as e:
                logger.warning(f"Failed to load {config_file}: {e}")
                continue
        
        # Limit multi-domain tasks if requested (paper uses 29, WebArena has 48)
        if max_multi_tasks and "multi" in subsets:
            multi_tasks = [t for t in self.tasks if t["subset"] == "multi"]
            if len(multi_tasks) > max_multi_tasks:
                logger.info(f"Limiting multi-domain tasks from {len(multi_tasks)} to {max_multi_tasks}")
                # Keep first N multi tasks (sorted by task ID for reproducibility)
                multi_tasks_sorted = sorted(multi_tasks, key=lambda x: x["numeric_id"])
                multi_tasks_to_keep = set(t["task_id"] for t in multi_tasks_sorted[:max_multi_tasks])
                self.tasks = [t for t in self.tasks if t["subset"] != "multi" or t["task_id"] in multi_tasks_to_keep]
        
        logger.info(f"Loaded {len(self.tasks)} tasks from WebArena")
        
        # Log statistics per subset
        subset_counts = {}
        for task in self.tasks:
            subset = task["subset"]
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        
        for subset, count in sorted(subset_counts.items()):
            expected = self.SUBSET_COUNTS.get(subset, "?")
            status = "✓" if count == expected else "⚠"
            logger.info(f"  {status} {subset}: {count} tasks (expected: {expected})")
        
        return self.tasks
    
    def _get_task_subset(self, config: Dict[str, Any]) -> str:
        """
        Determine task subset from config using WebArena's 'sites' field.
        
        WebArena naming:
        - shopping_admin = Admin/CMS subset (182 tasks)
        - shopping = E-commerce shopping (187 tasks)
        - gitlab = GitLab (180 tasks)
        - reddit = Reddit (106 tasks)
        - map = Map navigation (109 tasks)
        - multi-site combinations = Multi-domain (29 tasks in paper)
        
        Args:
            config: Task configuration dictionary
        
        Returns:
            Subset name (shopping, admin, gitlab, reddit, map, multi)
        """
        sites = config.get("sites") or []
        
        # Ensure sites is a list
        if not isinstance(sites, list):
            sites = [sites] if sites else []
        
        # Use the 'sites' field which is the authoritative source
        if not sites:
            logger.warning(f"No sites field in config, using URL fallback")
            # Fallback to URL if sites field missing
            start_url = config.get("start_url") or ""
            if not isinstance(start_url, str):
                start_url = str(start_url) if start_url else ""
            
            if ":7780" in start_url or "admin" in start_url:
                return "admin"
            elif ":7770" in start_url:
                return "shopping"
            elif ":8023" in start_url:
                return "gitlab"
            elif ":9999" in start_url:
                return "reddit"
            elif ":3000" in start_url:
                return "map"
            return "admin"
        
        # Multi-domain tasks (more than one site)
        if len(sites) > 1:
            return "multi"
        
        # Single-site tasks - map WebArena names to paper names
        site = sites[0] if sites else "shopping_admin"
        if not isinstance(site, str):
            site = str(site)
        
        # WebArena uses 'shopping_admin' for the Admin/CMS subset
        if site == "shopping_admin":
            return "admin"
        elif site == "shopping":
            return "shopping"
        elif site == "gitlab":
            return "gitlab"
        elif site == "reddit":
            return "reddit"
        elif site == "map":
            return "map"
        else:
            logger.warning(f"Unknown site '{site}', defaulting to 'admin'")
            return "admin"
    
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID."""
        for task in self.tasks:
            if task["task_id"] == task_id:
                return task
        return None
    
    def get_tasks_by_subset(self, subset: str) -> List[Dict[str, Any]]:
        """Get all tasks for a specific subset."""
        return [t for t in self.tasks if t["subset"] == subset]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        subset_counts = {}
        for task in self.tasks:
            subset = task["subset"]
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "subset_counts": subset_counts,
            "subsets": list(subset_counts.keys()),
        }
