"""
Utility to load evaluation specifications from WebArena config files.

This ensures we always use the correct, authoritative eval spec from the
data folder, rather than relying on potentially incomplete data in result files.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


class WebArenaEvalLoader:
    """Load evaluation specifications from WebArena config files."""
    
    def __init__(self, config_dir: str = "/Users/hivamoh/Desktop/ReasoningBank/data/webarena/config_files"):
        """
        Initialize loader.
        
        Args:
            config_dir: Path to WebArena config files directory
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            logger.warning(f"WebArena config directory not found: {config_dir}")
            self.config_dir = None
        else:
            logger.info(f"WebArena config loader initialized: {config_dir}")
    
    def extract_task_number(self, task_id: str) -> Optional[int]:
        """
        Extract numeric task ID from task_id string.
        
        Examples:
            "reddit_401" -> 401
            "shopping_123" -> 123
            "401" -> 401
        
        Args:
            task_id: Task identifier (e.g., "reddit_401")
        
        Returns:
            Numeric task ID or None if not found
        """
        # Try direct conversion first
        try:
            return int(task_id)
        except ValueError:
            pass
        
        # Extract number from string like "reddit_401"
        match = re.search(r'_(\d+)$', task_id)
        if match:
            return int(match.group(1))
        
        # Try finding any number in the string
        match = re.search(r'(\d+)', task_id)
        if match:
            return int(match.group(1))
        
        return None
    
    def load_eval_spec(self, task_id: str) -> Optional[Dict]:
        """
        Load evaluation specification for a task from WebArena config file.
        
        Args:
            task_id: Task identifier (e.g., "reddit_401")
        
        Returns:
            Eval spec dictionary or None if not found
        """
        if not self.config_dir:
            return None
        
        # Extract numeric task ID
        task_num = self.extract_task_number(task_id)
        if task_num is None:
            logger.warning(f"Could not extract task number from: {task_id}")
            return None
        
        # Load config file
        config_file = self.config_dir / f"{task_num}.json"
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            eval_spec = config.get('eval', {})
            
            if eval_spec:
                logger.debug(f"Loaded eval spec for task {task_id} (config {task_num}.json)")
                return eval_spec
            else:
                logger.warning(f"No eval spec in config file: {config_file}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return None
    
    def load_full_config(self, task_id: str) -> Optional[Dict]:
        """
        Load full task configuration from WebArena config file.
        
        Args:
            task_id: Task identifier (e.g., "reddit_401")
        
        Returns:
            Full config dictionary or None if not found
        """
        if not self.config_dir:
            return None
        
        task_num = self.extract_task_number(task_id)
        if task_num is None:
            return None
        
        config_file = self.config_dir / f"{task_num}.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return None
    
    def is_program_html_task(self, task_id: str) -> bool:
        """
        Check if a task uses program_html evaluation.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if task uses program_html evaluation
        """
        eval_spec = self.load_eval_spec(task_id)
        if not eval_spec:
            return False
        
        eval_types = eval_spec.get('eval_types', [])
        return 'program_html' in eval_types
    
    def get_reference_answer(self, task_id: str) -> Optional[str]:
        """
        Get reference answer for a task.
        
        For program_html tasks, extracts the required content.
        For regular tasks, extracts from reference_answers.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Reference answer string or None
        """
        eval_spec = self.load_eval_spec(task_id)
        if not eval_spec:
            return None
        
        # Check for program_html
        if 'program_html' in eval_spec.get('eval_types', []):
            program_html = eval_spec.get('program_html', [])
            for rule in program_html:
                required_contents = rule.get('required_contents', {})
                
                # Extract exact_match or must_include
                if 'exact_match' in required_contents:
                    return str(required_contents['exact_match'])
                elif 'must_include' in required_contents:
                    must_include = required_contents['must_include']
                    if isinstance(must_include, list):
                        return ', '.join(str(x) for x in must_include)
                    return str(must_include)
        
        # Check reference_answers
        reference_answers = eval_spec.get('reference_answers', {})
        if reference_answers:
            # Try different fields
            for field in ['exact_match', 'string_match', 'fuzzy_match']:
                if field in reference_answers:
                    value = reference_answers[field]
                    if isinstance(value, list) and value:
                        return str(value[0])
                    return str(value)
        
        return None


# Global instance
_loader = None

def get_eval_loader() -> WebArenaEvalLoader:
    """Get global WebArena eval loader instance."""
    global _loader
    if _loader is None:
        _loader = WebArenaEvalLoader()
    return _loader


def load_eval_spec_for_task(task_id: str) -> Optional[Dict]:
    """
    Convenience function to load eval spec for a task.
    
    Args:
        task_id: Task identifier (e.g., "reddit_401")
    
    Returns:
        Eval spec dictionary or None
    """
    loader = get_eval_loader()
    return loader.load_eval_spec(task_id)

