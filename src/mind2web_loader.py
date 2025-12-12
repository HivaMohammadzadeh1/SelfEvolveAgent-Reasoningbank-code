"""Mind2Web dataset loader for ReasoningBank.

Loads Mind2Web benchmark for web agent evaluation across real-world websites.
Based on: https://github.com/OSU-NLP-Group/Mind2Web
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datasets import load_dataset


# Websites with anti-bot protection that block BrowserGym
# Based on testing with browsergym/openended environment
BLOCKED_WEBSITES = {
    'seatgeek',    # DataDome CAPTCHA - only 14 nodes visible
    'delta',       # Anti-bot - 0 nodes visible
    'kohls',       # Anti-bot - 0 nodes visible
    'imdb',        # Cloudflare/bot detection - only 3 nodes
    'gamestop',    # Anti-bot - only 31 nodes
    'newegg',      # Anti-bot - only 40 nodes
    'viator',      # Anti-bot - only 27 nodes
    'carmax',      # Anti-bot (needs testing)
    'sixflags',    # Anti-bot (needs testing)
    'ticketcenter', # Anti-bot (needs testing)
}

# Websites confirmed to work well with BrowserGym (>50 interactive nodes)
WORKING_WEBSITES = {
    'budget',       # 823 nodes, 419 clickable
    'eventbrite',   # 977 nodes, 271 clickable
    'tvguide',      # 1129 nodes, 316 clickable
    'underarmour',  # 677 nodes, 553 clickable
    'soundcloud',   # 306 nodes, 110 clickable
}


class Mind2WebTask:
    """Single Mind2Web task instance."""

    def __init__(
        self,
        annotation_id: str,
        website: str,
        domain: str,
        subdomain: str,
        confirmed_task: str,
        action_reprs: List[str],
        actions: List[Dict[str, Any]],
        **kwargs
    ):
        self.annotation_id = annotation_id
        self.website = website
        self.domain = domain
        self.subdomain = subdomain
        self.confirmed_task = confirmed_task
        self.action_reprs = action_reprs  # Human-readable action descriptions
        self.actions = actions  # Full action annotations with target elements
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "website": self.website,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "confirmed_task": self.confirmed_task,
            "action_reprs": self.action_reprs,
            "actions": self.actions,
            **self.metadata
        }

    def get_task_description(self) -> str:
        """Get formatted task description for agent."""
        desc = f"Website: {self.website}\n"
        desc += f"Domain: {self.domain} / {self.subdomain}\n"
        desc += f"Task ID: {self.annotation_id}\n\n"
        desc += f"Task Goal:\n{self.confirmed_task}\n"
        return desc

    def get_num_steps(self) -> int:
        """Get number of action steps in ground truth."""
        return len(self.actions) if self.actions else 0


class Mind2WebDataset:
    """Mind2Web dataset loader.

    Mind2Web consists of 2,350 tasks from 137 websites across 31 domains.
    Three test splits for generalization:
    - Cross-Task (252 tasks): New tasks on seen websites/domains
    - Cross-Website (177 tasks): New websites within seen domains
    - Cross-Domain (912 tasks): New websites in unseen domains
    """

    # Test split sizes from Mind2Web paper
    SPLIT_COUNTS = {
        "test_task": 252,      # Cross-Task
        "test_website": 177,   # Cross-Website
        "test_domain": 912     # Cross-Domain
    }

    # Mapping from split names to paper names
    SPLIT_NAMES = {
        "test_task": "Cross-Task",
        "test_website": "Cross-Website",
        "test_domain": "Cross-Domain"
    }

    def __init__(
        self,
        split: str = "test_task",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Mind2Web dataset.

        Args:
            split: Dataset split to load. Options:
                  - "test_task": Cross-Task generalization (252 tasks)
                  - "test_website": Cross-Website generalization (177 tasks)
                  - "test_domain": Cross-Domain generalization (912 tasks)
            cache_dir: Optional cache directory for dataset
        """
        self.split = split
        self.cache_dir = cache_dir
        self.tasks: List[Mind2WebTask] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load Mind2Web dataset from HuggingFace or local files."""
        logger.info(f"Loading Mind2Web dataset: {self.split}...")

        # Check if loading from local files first (test splits are downloaded separately)
        if self.split in ["test_task", "test_website", "test_domain"]:
            # Try multiple possible paths
            possible_paths = [
                Path(f"data/mind2web/{self.split}"),
                Path(f"data/mind2web/test/{self.split}"),
            ]

            local_path = None
            for path in possible_paths:
                if path.exists():
                    local_path = path
                    break

            if local_path:
                logger.info(f"Loading from local path: {local_path}")
                self._load_from_local(local_path)
                return
            else:
                logger.warning(f"Local test data not found at {local_path}")
                logger.info("\n" + "="*60)
                logger.info("SETUP INSTRUCTIONS FOR MIND2WEB TEST SPLITS")
                logger.info("="*60)
                logger.info("The test splits are not directly available on HuggingFace to prevent data contamination.")
                logger.info("\nTo download the test splits:")
                logger.info("1. Visit: https://huggingface.co/datasets/osunlp/Mind2Web/tree/main")
                logger.info("2. Download the password-protected ZIP files for test splits")
                logger.info("3. Extract with password: mind2web")
                logger.info("4. Place the data in: data/mind2web/")
                logger.info("\nExpected structure:")
                logger.info("  data/mind2web/test_task/test_task.json")
                logger.info("  data/mind2web/test_website/test_website.json")
                logger.info("  data/mind2web/test_domain/test_domain.json")
                logger.info("="*60)
                raise FileNotFoundError(f"Test data not found at {local_path}. Please download from HuggingFace.")

        try:
            # Load training split from HuggingFace
            logger.info(f"Loading from HuggingFace: osunlp/Mind2Web")
            dataset = load_dataset(
                "osunlp/Mind2Web",
                split="train",  # Only train split is available on HuggingFace
                cache_dir=self.cache_dir
            )

            logger.info(f"Loaded {len(dataset)} instances from Mind2Web (train)")

            # Convert to Mind2WebTask objects
            for item in dataset:
                task = Mind2WebTask(**item)
                self.tasks.append(task)

            logger.info(f"Successfully loaded {len(self.tasks)} Mind2Web tasks")

            # Log domain distribution
            self._log_statistics()

        except Exception as e:
            logger.error(f"Failed to load Mind2Web: {e}")
            logger.error("Please install required packages: pip install datasets")
            raise

    def _load_from_local(self, data_path: Path):
        """Load Mind2Web test data from local JSON files."""
        try:
            # Find all JSON files in the directory (may be split across multiple files)
            json_files = sorted(data_path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {data_path}")

            logger.info(f"Found {len(json_files)} JSON file(s) in {data_path}")

            # Load all JSON files and combine
            for json_file in json_files:
                logger.info(f"Loading: {json_file.name}")

                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    # Handle both list format and single object format
                    if not isinstance(data, list):
                        data = [data]

                    # Convert to Mind2WebTask objects
                    for item in data:
                        try:
                            task = Mind2WebTask(**item)
                            self.tasks.append(task)
                        except Exception as e:
                            logger.warning(f"Failed to load task from {json_file.name}: {e}")
                            continue

                    logger.info(f"  Loaded {len(data)} tasks from {json_file.name}")

                except Exception as e:
                    logger.warning(f"Failed to load {json_file.name}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(self.tasks)} Mind2Web tasks from local files")

            # Log domain distribution
            self._log_statistics()

        except Exception as e:
            logger.error(f"Failed to load from local files: {e}")
            raise

    def _log_statistics(self):
        """Log dataset statistics."""
        domains = {}
        websites = set()
        for task in self.tasks:
            domains[task.domain] = domains.get(task.domain, 0) + 1
            websites.add(task.website)

        logger.info(f"Distribution: {len(websites)} unique websites, {len(domains)} domains")
        logger.info("Top domains:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {domain}: {count} tasks")

    def get_task(self, annotation_id: str) -> Optional[Mind2WebTask]:
        """Get task by annotation ID."""
        for task in self.tasks:
            if task.annotation_id == annotation_id:
                return task
        return None

    def get_tasks_by_domain(self, domain: str) -> List[Mind2WebTask]:
        """Get all tasks for a specific domain."""
        return [t for t in self.tasks if t.domain == domain]

    def get_tasks_by_website(self, website: str) -> List[Mind2WebTask]:
        """Get all tasks for a specific website."""
        return [t for t in self.tasks if t.website == website]

    def filter_tasks(
        self,
        domains: Optional[List[str]] = None,
        websites: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        start_index: int = 0,
        exclude_blocked: bool = False,
        only_working: bool = False
    ) -> List[Mind2WebTask]:
        """
        Filter tasks by domain/website and limit.

        Args:
            domains: List of domains to include (None = all)
            websites: List of websites to include (None = all)
            max_tasks: Maximum number of tasks to return
            start_index: Starting index for task selection (for parallel processing)
            exclude_blocked: If True, exclude websites with anti-bot protection
            only_working: If True, only include confirmed working websites

        Returns:
            Filtered list of tasks
        """
        tasks = self.tasks

        # Filter by BrowserGym compatibility
        if exclude_blocked:
            before_count = len(tasks)
            tasks = [t for t in tasks if t.website not in BLOCKED_WEBSITES]
            excluded = before_count - len(tasks)
            logger.info(f"Excluded {excluded} tasks from blocked websites (anti-bot protection)")
            logger.info(f"Remaining: {len(tasks)} tasks")

        if only_working:
            before_count = len(tasks)
            tasks = [t for t in tasks if t.website in WORKING_WEBSITES]
            filtered = before_count - len(tasks)
            logger.info(f"Filtered to {len(tasks)} tasks from {len(WORKING_WEBSITES)} working websites")
            logger.info(f"Excluded {filtered} tasks from untested/blocked websites")

        # Filter by domain
        if domains:
            tasks = [t for t in tasks if t.domain in domains]
            logger.info(f"Filtered to {len(tasks)} tasks from {len(domains)} domains")

        # Filter by website
        if websites:
            tasks = [t for t in tasks if t.website in websites]
            logger.info(f"Filtered to {len(tasks)} tasks from {len(websites)} websites")

        # Apply start_index and max_tasks for slicing
        if start_index > 0:
            tasks = tasks[start_index:]
            logger.info(f"Starting from index {start_index}")

        # Limit number of tasks
        if max_tasks and max_tasks < len(tasks):
            tasks = tasks[:max_tasks]
            logger.info(f"Limited to {max_tasks} tasks")

        return tasks

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Mind2WebTask:
        """Get task by index."""
        return self.tasks[idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        domains = {}
        websites = set()
        subdomains = {}

        for task in self.tasks:
            domains[task.domain] = domains.get(task.domain, 0) + 1
            websites.add(task.website)
            subdomains[task.subdomain] = subdomains.get(task.subdomain, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "num_websites": len(websites),
            "num_domains": len(domains),
            "num_subdomains": len(subdomains),
            "split": self.SPLIT_NAMES.get(self.split, self.split),
            "split_name": self.split,
            "expected_count": self.SPLIT_COUNTS.get(self.split, "?"),
            "domains": domains,
            "subdomains": subdomains
        }


def load_mind2web_dataset(
    split: str = "test_task",
    cache_dir: Optional[str] = None,
    domains: Optional[List[str]] = None,
    websites: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    start_index: int = 0,
    exclude_blocked: bool = False,
    only_working: bool = False
) -> Mind2WebDataset:
    """
    Convenience function to load Mind2Web dataset.

    Args:
        split: Dataset split ("test_task", "test_website", or "test_domain")
        cache_dir: Optional cache directory
        domains: Optional list of domains to filter
        websites: Optional list of websites to filter
        max_tasks: Optional maximum number of tasks
        start_index: Starting index for task selection (for parallel processing)
        exclude_blocked: If True, exclude websites with anti-bot protection
        only_working: If True, only include confirmed working websites

    Returns:
        Mind2WebDataset instance
    """
    dataset = Mind2WebDataset(split=split, cache_dir=cache_dir)

    if domains or websites or max_tasks or start_index > 0 or exclude_blocked or only_working:
        # Apply filtering and update dataset tasks
        filtered = dataset.filter_tasks(
            domains=domains,
            websites=websites,
            max_tasks=max_tasks,
            exclude_blocked=exclude_blocked,
            only_working=only_working,
            start_index=start_index
        )
        dataset.tasks = filtered
        logger.info(f"Dataset ready with {len(filtered)} tasks (start_index={start_index})")

    return dataset
