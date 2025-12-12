"""Concrete answer matching for accurate evaluation with improved fuzzy matching."""
import re
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from difflib import SequenceMatcher


class AnswerMatcher:
    """
    Concrete answer matching based on WebArena evaluation specifications.
    
    Enhanced with fuzzy matching to handle:
    - Format variations (natural language vs abbreviated)
    - Whitespace differences
    - Minor numeric differences
    - Multi-line answers
    
    Supports different evaluation types:
    - exact_match: Exact string match (case-insensitive)
    - string_match: Substring or fuzzy match
    - must_include: All required strings must be present
    - must_exclude: None of the excluded strings should be present
    """
    
    def __init__(self, numeric_tolerance: float = 0.02):
        """
        Initialize answer matcher.
        
        Args:
            numeric_tolerance: Tolerance for numeric comparisons (default 2% for distances)
                - Allows "455km" to match "457km" (difference: 0.4%)
                - Allows "914km" to match "929km" (difference: 1.6%)
                - But rejects wildly different values (>2%)
        """
        self.numeric_tolerance = numeric_tolerance
    
    def evaluate_answer(
        self,
        agent_answer: str,
        task_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evaluate agent's answer against reference answer from task data.
        
        Args:
            agent_answer: Final answer from agent
            task_data: Task data containing eval specifications
        
        Returns:
            (success: bool, explanation: str)
        """
        if not agent_answer or not agent_answer.strip():
            return False, "Agent provided no answer"
        
        # Extract evaluation specs from task data
        eval_spec = task_data.get("eval", {})
        if not eval_spec:
            logger.warning(f"No eval spec found in task {task_data.get('task_id')}")
            # Fall back to using ground_truth if available
            if "ground_truth" in task_data:
                return self._evaluate_simple_match(agent_answer, task_data["ground_truth"])
            return False, "No evaluation specification found"
        
        eval_types = eval_spec.get("eval_types", [])
        reference_answers = eval_spec.get("reference_answers", {})
        
        # Check for program_html evaluation type
        if "program_html" in eval_types:
            # Extract reference from program_html if reference_answers is null
            if not reference_answers or reference_answers is None:
                return self._evaluate_program_html(agent_answer, eval_spec, task_data.get('task_id'))
        
        if not eval_types or not reference_answers:
            logger.warning(f"Incomplete eval spec in task {task_data.get('task_id')}")
            # Try ground_truth as fallback
            if "ground_truth" in task_data:
                return self._evaluate_simple_match(agent_answer, task_data["ground_truth"])
            return False, "Incomplete evaluation specification"
        
        # Normalize agent answer
        agent_answer_normalized = self._normalize_string(agent_answer)
        
        # Try each evaluation type
        for eval_type in eval_types:
            if eval_type == "exact_match":
                success, explanation = self._evaluate_exact_match(
                    agent_answer_normalized,
                    reference_answers,
                    agent_answer
                )
                if success:
                    return True, explanation
            
            elif eval_type == "string_match":
                success, explanation = self._evaluate_string_match(
                    agent_answer_normalized,
                    reference_answers,
                    agent_answer
                )
                if success:
                    return True, explanation
            
            elif eval_type == "must_include":
                success, explanation = self._evaluate_must_include(
                    agent_answer_normalized,
                    reference_answers,
                    agent_answer
                )
                if success:
                    return True, explanation
            
            elif eval_type == "must_exclude":
                success, explanation = self._evaluate_must_exclude(
                    agent_answer_normalized,
                    reference_answers,
                    agent_answer
                )
                if not success:  # For must_exclude, failure means exclusion violated
                    return False, explanation
        
        # If we get here, none of the eval types matched
        return False, f"Answer '{agent_answer[:100]}' did not match any reference answer"
    
    def _normalize_string(self, s: str) -> str:
        """Normalize string for comparison: lowercase, trim, normalize whitespace."""
        if not s:
            return ""
        # Lowercase
        s = s.lower()
        # Normalize whitespace (including removing spaces around units)
        s = re.sub(r'\s+', ' ', s)
        # Trim
        s = s.strip()
        return s
    
    def _normalize_time_format(self, s: str) -> str:
        """
        Normalize time formats to handle variations like:
        - "1 hour and 23 minutes" -> "1h 23min"
        - "10 hours and 33 minutes" -> "10h 33min"
        """
        # Pattern: X hour(s) and Y minute(s) -> Xh Ymin
        s = re.sub(r'(\d+)\s*hours?\s+and\s+(\d+)\s*minutes?', r'\1h \2min', s)
        s = re.sub(r'(\d+)\s*hours?', r'\1h', s)
        s = re.sub(r'(\d+)\s*minutes?', r'\1min', s)
        # Normalize spacing: "1h23min" -> "1h 23min"
        s = re.sub(r'(\d+h)(\d+min)', r'\1 \2', s)
        return s
    
    def _normalize_distance_format(self, s: str) -> str:
        """
        Normalize distance formats to handle variations:
        - "914km" -> "914km"
        - "914 km" -> "914km"
        - "455 kilometers" -> "455km"
        
        Strategy: Remove all spaces between numbers and units for consistent comparison
        """
        # First expand long forms to short forms
        s = re.sub(r'kilometers?', 'km', s, flags=re.IGNORECASE)
        s = re.sub(r'miles?(?!\w)', 'mi', s, flags=re.IGNORECASE)  # "miles" but not "miles" in middle of word
        s = re.sub(r'meters?(?!\w)', 'm', s, flags=re.IGNORECASE)
        
        # Remove spaces between numbers and units: "914 km" -> "914km"
        s = re.sub(r'(\d+)\s*(km|mi|m)', r'\1\2', s, flags=re.IGNORECASE)
        
        return s
    
    def _extract_numbers(self, s: str) -> List[float]:
        """Extract all numbers from a string."""
        # Match integers and floats
        numbers = re.findall(r'\d+\.?\d*', s)
        return [float(n) for n in numbers]
    
    def _numbers_match(self, numbers1: List[float], numbers2: List[float]) -> bool:
        """
        Check if two lists of numbers match within tolerance.
        """
        if len(numbers1) != len(numbers2):
            return False
        
        for n1, n2 in zip(numbers1, numbers2):
            # Check if within tolerance (percentage or absolute)
            if n2 == 0:
                if abs(n1 - n2) > 0.1:  # Absolute tolerance for small numbers
                    return False
            else:
                relative_diff = abs(n1 - n2) / max(abs(n1), abs(n2))
                if relative_diff > self.numeric_tolerance:
                    return False
        
        return True
    
    def _fuzzy_string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings (0-1)."""
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _evaluate_simple_match(self, agent_answer: str, reference: str) -> Tuple[bool, str]:
        """Simple fallback evaluation when no eval spec is available."""
        agent_norm = self._normalize_string(agent_answer)
        ref_norm = self._normalize_string(str(reference))
        
        # Exact match
        if agent_norm == ref_norm:
            return True, f"Exact match: '{agent_answer}' == '{reference}'"
        
        # Substring match
        if ref_norm in agent_norm or agent_norm in ref_norm:
            return True, f"Substring match: '{agent_answer}' contains '{reference}'"
        
        # High similarity match
        similarity = self._fuzzy_string_similarity(agent_norm, ref_norm)
        if similarity > 0.85:
            return True, f"High similarity match ({similarity:.2%}): '{agent_answer}' ≈ '{reference}'"
        
        return False, f"No match: '{agent_answer}' vs '{reference}'"
    
    def _evaluate_exact_match(
        self,
        agent_answer: str,
        reference_answers: Dict[str, Any],
        original_answer: str
    ) -> Tuple[bool, str]:
        """Evaluate exact match with enhanced fuzzy logic."""
        # Handle None reference_answers
        if reference_answers is None:
            return False, "No reference answers provided"
        
        # Try multiple possible field names in WebArena data
        expected = None
        for field in ["exact_match", "fuzzy_match", "string_match", "reference_answer_raw_annotation"]:
            if field in reference_answers:
                expected = reference_answers[field]
                break
        
        if not expected:
            return False, "No exact_match reference found"
        
        # Handle list format
        if isinstance(expected, list):
            if not expected:
                return False, "Empty reference answer list"
            expected_list = expected
        else:
            expected_list = [expected]
        
        # Try matching against each expected answer
        for exp in expected_list:
            exp_str = str(exp)
            exp_normalized = self._normalize_string(exp_str)
            
            # 1. Exact match
            if agent_answer == exp_normalized:
                return True, f"Exact match: '{original_answer}' == '{exp_str}'"
            
            # 2. Match after time format normalization
            agent_time_norm = self._normalize_time_format(agent_answer)
            exp_time_norm = self._normalize_time_format(exp_normalized)
            if agent_time_norm == exp_time_norm:
                return True, f"Time format match: '{original_answer}' == '{exp_str}'"
            
            # 3. Match after distance format normalization
            agent_dist_norm = self._normalize_distance_format(agent_answer)
            exp_dist_norm = self._normalize_distance_format(exp_normalized)
            if agent_dist_norm == exp_dist_norm:
                return True, f"Distance format match: '{original_answer}' == '{exp_str}'"
            
            # 4. Fuzzy numeric match for distances (e.g., 455km vs 457km)
            # Extract numbers from both and check if they're within tolerance
            agent_numbers = self._extract_numbers(agent_dist_norm)
            exp_numbers = self._extract_numbers(exp_dist_norm)
            if agent_numbers and exp_numbers and self._numbers_match(agent_numbers, exp_numbers):
                # Also check that units match (km vs km, not km vs mi)
                agent_units = re.findall(r'(km|mi|m)(?!\w)', agent_dist_norm, re.IGNORECASE)
                exp_units = re.findall(r'(km|mi|m)(?!\w)', exp_dist_norm, re.IGNORECASE)
                if agent_units and exp_units and agent_units[0].lower() == exp_units[0].lower():
                    return True, f"Fuzzy distance match: '{original_answer}' ≈ '{exp_str}' (within {self.numeric_tolerance:.1%} tolerance)"
            
            # 5. Substring match for exact_match (agent gave more context)
            # Check if normalized reference is in normalized agent answer
            if exp_normalized in agent_answer:
                return True, f"Exact content match: '{exp_str}' found in '{original_answer}'"
            if exp_dist_norm in agent_dist_norm:
                return True, f"Distance-normalized match: '{exp_str}' found in '{original_answer}'"
            if exp_time_norm in agent_time_norm:
                return True, f"Time-normalized match: '{exp_str}' found in '{original_answer}'"
            
            # 5. Numeric match with tolerance
            agent_numbers = self._extract_numbers(agent_answer)
            exp_numbers = self._extract_numbers(exp_normalized)
            if agent_numbers and exp_numbers and self._numbers_match(agent_numbers, exp_numbers):
                # Check if the text context is similar
                agent_text = re.sub(r'\d+\.?\d*', '', agent_answer).strip()
                exp_text = re.sub(r'\d+\.?\d*', '', exp_normalized).strip()
                text_sim = self._fuzzy_string_similarity(agent_text, exp_text)
                if text_sim > 0.6:
                    return True, f"Numeric match with similar context: '{original_answer}' ≈ '{exp_str}'"
            
            # 6. High string similarity
            similarity = self._fuzzy_string_similarity(agent_answer, exp_normalized)
            if similarity > 0.90:
                return True, f"High similarity match ({similarity:.2%}): '{original_answer}' ≈ '{exp_str}'"
        
        return False, f"No exact match: '{original_answer}' not in {expected_list}"
    
    def _evaluate_string_match(
        self,
        agent_answer: str,
        reference_answers: Dict[str, Any],
        original_answer: str
    ) -> Tuple[bool, str]:
        """
        Evaluate string match with improved fuzzy matching.
        
        Checks:
        1. Exact match
        2. Substring match
        3. Time/distance format normalization
        4. Numeric match with tolerance
        5. Word overlap
        6. Multi-line answer handling
        """
        # Handle None reference_answers
        if reference_answers is None:
            return False, "No reference answers provided"
        
        # Try multiple possible field names in WebArena data
        expected = None
        for field in ["exact_match", "string_match", "fuzzy_match", "reference_answer_raw_annotation"]:
            if field in reference_answers:
                expected = reference_answers[field]
                break
        
        if not expected:
            return False, "No string_match reference found"
        
        # Handle list format
        if isinstance(expected, list):
            if not expected:
                return False, "Empty reference answer list"
            expected_list = expected
        else:
            expected_list = [expected]
        
        # Try matching against each expected answer
        for exp in expected_list:
            exp_str = str(exp)
            exp_normalized = self._normalize_string(exp_str)
            
            # 1. Exact match
            if agent_answer == exp_normalized:
                return True, f"Exact match: '{original_answer}' == '{exp_str}'"
            
            # 2. Substring match (both directions)
            if exp_normalized in agent_answer:
                return True, f"Substring match: '{exp_str}' found in '{original_answer}'"
            if agent_answer in exp_normalized:
                return True, f"Substring match: '{original_answer}' is part of '{exp_str}'"
            
            # 3. Handle multi-line answers (split by newline)
            if '\n' in exp_str:
                exp_parts = [self._normalize_string(p) for p in exp_str.split('\n') if p.strip()]
                agent_answer_numbers = self._extract_numbers(agent_answer)
                matches = []
                
                for part in exp_parts:
                    # Check substring match
                    if part in agent_answer or agent_answer in part:
                        matches.append(part)
                        continue
                    
                    # Check with time normalization
                    part_time = self._normalize_time_format(part)
                    agent_time = self._normalize_time_format(agent_answer)
                    if part_time in agent_time or agent_time in part_time:
                        matches.append(part)
                        continue
                    
                    # Check with distance normalization
                    part_dist = self._normalize_distance_format(part)
                    agent_dist = self._normalize_distance_format(agent_answer)
                    if part_dist in agent_dist or agent_dist in part_dist:
                        matches.append(part)
                        continue
                    
                    # Check numeric match with tolerance for this part
                    part_numbers = self._extract_numbers(part)
                    if part_numbers and agent_answer_numbers:
                        # Check if any of the agent numbers match any of the part numbers
                        for pnum in part_numbers:
                            for anum in agent_answer_numbers:
                                # Allow 10% tolerance for distance/time measurements
                                if anum == 0 and pnum == 0:
                                    matches.append(part)
                                    break
                                elif pnum == 0:
                                    if abs(anum - pnum) < 1:
                                        matches.append(part)
                                        break
                                else:
                                    relative_diff = abs(anum - pnum) / max(abs(anum), abs(pnum))
                                    if relative_diff <= 0.10:  # 10% tolerance
                                        matches.append(part)
                                        break
                            if part in matches:
                                break
                
                # Success if all parts match OR if it's a 2-part answer and both key components present
                if len(matches) >= len(exp_parts):
                    return True, f"Multi-part match: all {len(exp_parts)} parts found in '{original_answer}'"
                elif len(matches) >= len(exp_parts) * 0.5:  # At least 50% of parts match
                    return True, f"Multi-part match: {len(matches)}/{len(exp_parts)} parts found in '{original_answer}'"
            
            # 4. Time format normalization
            agent_time_norm = self._normalize_time_format(agent_answer)
            exp_time_norm = self._normalize_time_format(exp_normalized)
            if agent_time_norm == exp_time_norm:
                return True, f"Time format match: '{original_answer}' == '{exp_str}'"
            if exp_time_norm in agent_time_norm or agent_time_norm in exp_time_norm:
                return True, f"Time format substring match: '{original_answer}' contains '{exp_str}'"
            
            # 5. Distance format normalization
            agent_dist_norm = self._normalize_distance_format(agent_answer)
            exp_dist_norm = self._normalize_distance_format(exp_normalized)
            if agent_dist_norm == exp_dist_norm:
                return True, f"Distance format match: '{original_answer}' == '{exp_str}'"
            if exp_dist_norm in agent_dist_norm or agent_dist_norm in exp_dist_norm:
                return True, f"Distance format substring match: '{original_answer}' contains '{exp_str}'"
            
            # 6. Numeric match with tolerance
            agent_numbers = self._extract_numbers(agent_answer)
            exp_numbers = self._extract_numbers(exp_normalized)
            if agent_numbers and exp_numbers and self._numbers_match(agent_numbers, exp_numbers):
                return True, f"Numeric match: numbers in '{original_answer}' match '{exp_str}' within tolerance"
            
            # 7. Word overlap for short references
            exp_words = set(exp_normalized.split())
            agent_words = set(agent_answer.split())
            
            if len(exp_words) <= 5:
                overlap = exp_words.intersection(agent_words)
                if len(overlap) >= len(exp_words):
                    return True, f"All key words from '{exp_str}' found in '{original_answer}'"
            
            # 8. Percentage word overlap for longer references
            if len(exp_words) > 5:
                overlap = exp_words.intersection(agent_words)
                overlap_ratio = len(overlap) / len(exp_words) if exp_words else 0
                if overlap_ratio > 0.6:
                    return True, f"Word overlap match ({overlap_ratio:.0%}): '{original_answer}' vs '{exp_str}'"
            
            # 9. High overall similarity
            similarity = self._fuzzy_string_similarity(agent_answer, exp_normalized)
            if similarity > 0.75:
                return True, f"High similarity match ({similarity:.2%}): '{original_answer}' ≈ '{exp_str}'"
        
        return False, f"No string match: '{original_answer}' vs {expected_list}"
    
    def _evaluate_must_include(
        self,
        agent_answer: str,
        reference_answers: Dict[str, Any],
        original_answer: str
    ) -> Tuple[bool, str]:
        """Evaluate must_include: all required strings must be present."""
        # Handle None reference_answers
        if reference_answers is None:
            return False, "No reference answers provided"
        
        required = reference_answers.get("must_include", [])
        if not required:
            return False, "No must_include references found"
        
        if isinstance(required, str):
            required = [required]
        
        missing = []
        for req in required:
            req_normalized = self._normalize_string(str(req))
            
            # Check direct substring
            if req_normalized in agent_answer:
                continue
            
            # Check with format normalization
            req_time = self._normalize_time_format(req_normalized)
            agent_time = self._normalize_time_format(agent_answer)
            if req_time in agent_time:
                continue
            
            req_dist = self._normalize_distance_format(req_normalized)
            agent_dist = self._normalize_distance_format(agent_answer)
            if req_dist in agent_dist:
                continue
            
            # If still not found, mark as missing
            missing.append(req)
        
        if not missing:
            return True, f"All required strings found in '{original_answer}'"
        
        return False, f"Missing required strings: {missing}"
    
    def _evaluate_must_exclude(
        self,
        agent_answer: str,
        reference_answers: Dict[str, Any],
        original_answer: str
    ) -> Tuple[bool, str]:
        """Evaluate must_exclude: none of the excluded strings should be present."""
        # Handle None reference_answers
        if reference_answers is None:
            return True, "No reference answers provided (pass by default)"
        
        excluded = reference_answers.get("must_exclude", [])
        if not excluded:
            return True, "No must_exclude references (pass by default)"
        
        if isinstance(excluded, str):
            excluded = [excluded]
        
        found = []
        for exc in excluded:
            exc_normalized = self._normalize_string(str(exc))
            if exc_normalized in agent_answer:
                found.append(exc)
        
        if not found:
            return True, f"No excluded strings found in '{original_answer}'"
        
        return False, f"Found excluded strings: {found}"
    
    def _evaluate_program_html(
        self,
        agent_answer: str,
        eval_spec: Dict[str, Any],
        task_id: str
    ) -> Tuple[bool, str]:
        """
        Evaluate based on program_html specification.
        
        For program_html tasks, we check if the agent's answer (usually a URL or page content)
        contains the required content specified in the program_html rules.
        """
        program_html = eval_spec.get("program_html", [])
        if not program_html:
            return False, "No program_html specification found"
        
        # Extract required contents from program_html
        for rule in program_html:
            required_contents = rule.get("required_contents", {})
            must_include = required_contents.get("must_include", [])
            must_exclude = required_contents.get("must_exclude", [])
            
            if must_include:
                # Check if all required strings are in the answer
                agent_normalized = self._normalize_string(agent_answer)
                missing = []
                for req in must_include:
                    req_normalized = self._normalize_string(str(req))
                    if req_normalized not in agent_normalized:
                        missing.append(req)
                
                if missing:
                    return False, f"Answer missing required content: {missing}"
                
                # Also check must_exclude if specified
                if must_exclude:
                    found_excluded = []
                    for exc in must_exclude:
                        exc_normalized = self._normalize_string(str(exc))
                        if exc_normalized in agent_normalized:
                            found_excluded.append(exc)
                    
                    if found_excluded:
                        return False, f"Answer contains excluded content: {found_excluded}"
                
                return True, f"Answer contains all required content from program_html: {must_include}"
        
        # If no must_include rules found, we can't evaluate
        return False, "No evaluatable program_html rules found"
    
    def get_reference_answer(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Extract the reference answer from task data for logging."""
        eval_spec = task_data.get("eval", {})
        if not eval_spec:
            # Fallback to ground_truth
            return task_data.get("ground_truth")
        
        reference_answers = eval_spec.get("reference_answers", {})
        
        # Handle None reference_answers - check for program_html
        if reference_answers is None:
            # Try to extract from program_html
            eval_types = eval_spec.get("eval_types", [])
            if "program_html" in eval_types:
                program_html = eval_spec.get("program_html", [])
                for rule in program_html:
                    required_contents = rule.get("required_contents", {})
                    must_include = required_contents.get("must_include", [])
                    if must_include:
                        if isinstance(must_include, list):
                            return ", ".join(str(i) for i in must_include)
                        return str(must_include)
            
            # Check reference_answer_raw_annotation
            if "reference_answer_raw_annotation" in eval_spec:
                return str(eval_spec["reference_answer_raw_annotation"])
            
            return task_data.get("ground_truth")
        
        # Try different fields in order of preference
        for field in ["exact_match", "string_match", "fuzzy_match", "reference_answer_raw_annotation"]:
            if field in reference_answers:
                value = reference_answers[field]
                # Handle list format
                if isinstance(value, list):
                    return str(value[0]) if value else None
                return str(value)
            if field in eval_spec:
                value = eval_spec[field]
                if isinstance(value, list):
                    return str(value[0]) if value else None
                return str(value)
        
        # Try must_include as last resort
        if "must_include" in reference_answers:
            includes = reference_answers["must_include"]
            if isinstance(includes, list):
                return ", ".join(str(i) for i in includes)
            return str(includes)
        
        # Final fallback to ground_truth
        return task_data.get("ground_truth")
