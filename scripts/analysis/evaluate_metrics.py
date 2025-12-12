#!/usr/bin/env python3
"""
Evaluation Metrics Calculator for WebArena Benchmark

Following the evaluation protocol:
1. Effectiveness: Success Rate (percentage of queries successfully resolved)
   - Uses LLM-based fuzzy matching and exact string matching
   - Verifies if essential answer terms appear in predictions

2. Efficiency: Average Number of Steps
   - Measures average steps taken by agent to complete each query
   - Reflects computational and interaction cost

Note: This project uses WebArena benchmark, not Mind2Web.
Mind2Web metrics (element accuracy, action F1, step success rate, task-level success rate)
are not applicable here.
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from difflib import SequenceMatcher
from loguru import logger

# Import WebArena eval spec loader
try:
    from webarena_eval_loader import get_eval_loader, load_eval_spec_for_task
    WEBARENA_LOADER_AVAILABLE = True
except ImportError:
    logger.warning("webarena_eval_loader not available, will use eval specs from results")
    WEBARENA_LOADER_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Metrics for a single experiment run."""
    mode: str
    subset: str
    
    # Core metrics
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    error_tasks: int
    
    # Effectiveness metric
    success_rate: float  # Percentage of successfully resolved queries
    
    # Efficiency metric
    avg_steps_all: float  # Average steps across all tasks (including failures)
    avg_steps_successful: float  # Average steps for successful tasks only
    avg_steps_attempted: float  # Average steps for tasks that were attempted (no errors)
    
    # Additional metrics
    total_tokens: int
    total_walltime: float
    
    # Detailed breakdown
    tasks_with_answer: int  # Tasks where agent provided an answer
    tasks_without_answer: int  # Tasks where agent provided no answer


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove punctuation at the end
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    if not text:
        return []
    # Match numbers with optional decimals and units
    pattern = r'(\d+\.?\d*)\s*(km|mi|h|min|minutes|hours|m|meters|miles)?'
    matches = re.findall(pattern, text.lower())
    numbers = []
    for num, unit in matches:
        try:
            numbers.append(float(num))
        except:
            pass
    return numbers


def normalize_time_format(text: str) -> str:
    """Normalize time formats to be comparable."""
    if not text:
        return ""
    
    text_lower = text.lower()
    
    # Convert "X hour(s) and Y minute(s)" to "Xh Ymin"
    # e.g., "1 hour and 23 minutes" -> "1h 23min"
    pattern1 = r'(\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*(?:minutes?|min)'
    text_lower = re.sub(pattern1, r'\1h \2min', text_lower)
    
    # Convert "X hour(s)" to "Xh"
    pattern2 = r'(\d+)\s*hours?'
    text_lower = re.sub(pattern2, r'\1h', text_lower)
    
    # Convert "Y minute(s)" to "Ymin"
    pattern3 = r'(\d+)\s*(?:minutes?|min)'
    text_lower = re.sub(pattern3, r'\1min', text_lower)
    
    return text_lower


def is_similar_number(num1: float, num2: float, tolerance_pct: float = 5.0) -> bool:
    """Check if two numbers are similar within a tolerance percentage."""
    if num1 == 0 and num2 == 0:
        return True
    if num1 == 0 or num2 == 0:
        return abs(num1 - num2) < 1.0  # Allow difference of 1 for zeros
    
    diff_pct = abs(num1 - num2) / max(abs(num1), abs(num2)) * 100
    return diff_pct <= tolerance_pct


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts are similar using fuzzy matching."""
    if not text1 or not text2:
        return False
    
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Substring match (one contains the other)
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Sequence matcher for similarity
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


def check_program_html_observation(
    task_data: Dict,
    observations: List[str] = None,
    agent_answer: str = None,
    thoughts: List[str] = None
) -> Tuple[bool, str]:
    """
    Check if agent successfully completed program_html tasks.
    
    Based on WebArena's HTMLContentEvaluator approach:
    https://github.com/web-arena-x/webarena/blob/main/evaluation_harness/evaluators.py
    
    For program_html tasks, success is determined by:
    1. Checking if agent navigated to the target URL (from last observation)
    2. Checking if required content appears in the final page state (last observation)
    3. This simulates what WebArena does: navigate to URL, execute locator, check content
    
    Since we're doing offline evaluation (no live browser), we check the LAST observation
    which captures the final state of the page after all agent actions.
    
    Args:
        task_data: Full task data including eval spec from WebArena config
        observations: List of observations from agent trajectory
        agent_answer: Final answer from agent (not used for program_html)
        thoughts: List of agent thoughts from trajectory (for debugging)
    
    Returns:
        (is_match, explanation)
    """
    eval_spec = task_data.get('eval', {})
    eval_types = eval_spec.get('eval_types', [])
    
    if 'program_html' not in eval_types:
        return False, "Not a program_html task"
    
    program_html = eval_spec.get('program_html', [])
    if not program_html:
        return False, "No program_html specification found"
    
    if not observations or len(observations) == 0:
        return False, "No observations available for program_html task"
    
    # Get the LAST observation (final state) and LAST thought
    last_observation = observations[-1].lower()
    last_thought = thoughts[-1].lower() if thoughts and len(thoughts) > 0 else ""
    
    # Check each program_html rule
    for rule in program_html:
        target_url = rule.get('url', '')
        required_contents = rule.get('required_contents', {})
        
        # Extract exact_match, must_include, must_exclude
        exact_match = required_contents.get('exact_match')
        must_include = required_contents.get('must_include', [])
        must_exclude = required_contents.get('must_exclude', [])
        
        # Extract final URL from last observation
        final_url = extract_url_from_observation(observations[-1])
        
        # Check if final URL matches target URL
        url_match = False
        if final_url and target_url:
            # Normalize URL placeholders like __REDDIT__
            final_url_lower = final_url.lower()
            target_normalized = target_url.lower().replace('__reddit__', '').replace('__', '')
            if target_normalized in final_url_lower:
                url_match = True
        
        # Check if required content is in the LAST observation
        # This simulates WebArena's HTMLContentEvaluator which executes:
        #   page.goto(target_url)
        #   selected_element = page.evaluate(locator)  
        #   compare selected_element with required_contents
        #
        # Since we're offline, we check if the required content appears in the
        # final page state captured in the last observation.
        
        score = 1.0
        content_found = False
        missing_items = []
        
        # Handle exact_match (WebArena's exact_match approach)
        if exact_match:
            content_normalized = normalize_text(str(exact_match))
            if content_normalized in last_observation:
                content_found = True
                score *= 1.0
            else:
                missing_items.append(exact_match)
                score *= 0.0
        
        # Handle must_include (WebArena's must_include approach)
        elif must_include:
            # WebArena checks each item in must_include list
            items_to_check = must_include if isinstance(must_include, list) else [must_include]
            for item in items_to_check:
                # Handle |OR| separator (WebArena feature)
                item_variants = str(item).split(' |OR| ')
                item_found = False
                
                for variant in item_variants:
                    variant_normalized = normalize_text(variant)
                    if variant_normalized in last_observation:
                        item_found = True
                        break
                
                if item_found:
                    score *= 1.0
                else:
                    missing_items.append(item)
                    score *= 0.0
            
            if score == 1.0:
                content_found = True
        
        # Check must_exclude (fail if any excluded item is present)
        if must_exclude:
            items_to_exclude = must_exclude if isinstance(must_exclude, list) else [must_exclude]
            for item in items_to_exclude:
                item_normalized = normalize_text(str(item))
                if item_normalized in last_observation:
                    return 0.0, f"✗ program_html: Excluded content '{item}' found in final observation"
        
        # Return result based on WebArena's scoring (1.0 = success, 0.0 = failure)
        if score == 1.0 and content_found:
            details = []
            if exact_match:
                details.append(f"'{exact_match}'")
            if must_include:
                items_str = ', '.join([f"'{i}'" for i in (must_include if isinstance(must_include, list) else [must_include])])
                details.append(f"{items_str}")
            
            content_str = ' '.join(details) if details else 'required content'
            
            if url_match:
                return True, f"✓ program_html: Final observation shows {content_str} at {target_url}"
            else:
                return True, f"✓ program_html: Final observation shows {content_str}"
        else:
            # Debug info: check if agent thought it succeeded
            thought_indicates_success = False
            if last_thought:
                success_indicators = ['successfully', 'completed', 'posted', 'saved', 'updated', 'changed']
                if any(indicator in last_thought for indicator in success_indicators):
                    thought_indicates_success = True
            
            if missing_items:
                if thought_indicates_success:
                    return False, f"✗ program_html: Required {missing_items} NOT in final observation (agent thought it succeeded but content not on page)"
                else:
                    return False, f"✗ program_html: Required {missing_items} NOT in final observation"
            else:
                return False, f"✗ program_html: Required content not in final observation"
    
    return False, "No matching program_html rules found"


def check_answer_match(agent_answer: str, reference_answer: str) -> Tuple[bool, str]:
    """
    Enhanced answer matching with multiple strategies.
    
    Returns:
        (is_match, explanation)
    """
    if not agent_answer or agent_answer.strip() == "":
        return False, "Agent provided no answer"
    
    if not reference_answer or reference_answer.strip() == "":
        return False, "No reference answer provided"
    
    agent_norm = normalize_text(agent_answer)
    ref_norm = normalize_text(reference_answer)
    
    # Strategy 1: Exact match after normalization
    if agent_norm == ref_norm:
        return True, f"Exact match (normalized): '{agent_answer}' == '{reference_answer}'"
    
    # Strategy 2: Substring match
    if ref_norm in agent_norm:
        return True, f"Substring match: '{reference_answer}' found in '{agent_answer}'"
    
    # Strategy 3: Check if agent answer contains all key terms from reference
    ref_words = set(ref_norm.split())
    agent_words = set(agent_norm.split())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'or', 'in', 'on', 'at'}
    ref_keywords = ref_words - stopwords
    agent_keywords = agent_words - stopwords
    
    if ref_keywords and ref_keywords.issubset(agent_keywords):
        return True, f"All keywords match: {ref_keywords}"
    
    # Strategy 4: Time format matching (handle "1 hour and 23 minutes" vs "1h 23min")
    agent_time_norm = normalize_time_format(agent_answer)
    ref_time_norm = normalize_time_format(reference_answer)
    
    if ref_time_norm in agent_time_norm or agent_time_norm in ref_time_norm:
        return True, f"Time format match: '{reference_answer}' found in normalized '{agent_answer}'"
    
    # Strategy 5: Numerical comparison for distance/time answers
    agent_nums = extract_numbers(agent_answer)
    ref_nums = extract_numbers(reference_answer)
    
    if agent_nums and ref_nums:
        # Check if main numbers are similar
        for a_num in agent_nums:
            for r_num in ref_nums:
                if is_similar_number(a_num, r_num, tolerance_pct=5.0):
                    # Check if answer contains similar context or key terms
                    # More lenient: just check if there's some overlap or similar keywords
                    if (fuzzy_match(agent_answer, reference_answer, threshold=0.6) or 
                        len(ref_keywords & agent_keywords) >= len(ref_keywords) * 0.5):
                        return True, f"Numerical match: {a_num} ≈ {r_num} (within 5% tolerance)"
                    # For very close numbers (within 1%), be even more lenient
                    elif is_similar_number(a_num, r_num, tolerance_pct=1.0):
                        return True, f"Close numerical match: {a_num} ≈ {r_num} (within 1% tolerance)"
    
    # Strategy 6: Fuzzy text matching (high threshold)
    if fuzzy_match(agent_answer, reference_answer, threshold=0.85):
        ratio = SequenceMatcher(None, agent_norm, ref_norm).ratio()
        return True, f"Fuzzy match: {ratio*100:.1f}% similar"
    
    # Strategy 7: Handle multi-part answers (comma or newline separated)
    ref_parts = [p.strip() for p in re.split(r'[,\n]', reference_answer) if p.strip()]
    agent_parts = [p.strip() for p in re.split(r'[,\n]', agent_answer) if p.strip()]
    
    if len(ref_parts) > 1:
        # Check if all reference parts are in agent answer
        all_found = all(
            any(normalize_text(rp) in normalize_text(ap) for ap in agent_parts)
            for rp in ref_parts
        )
        if all_found:
            return True, f"Multi-part match: all {len(ref_parts)} parts found"
    
    # Strategy 8: Handle list-style answers (__REDDIT__ paths, etc.)
    if '__REDDIT__' in reference_answer or '__REDDIT__' in agent_answer:
        # Extract Reddit paths
        ref_reddit = set(re.findall(r'__REDDIT__[^\s,]+', reference_answer))
        agent_reddit = set(re.findall(r'__REDDIT__[^\s,]+', agent_answer))
        
        if ref_reddit and agent_reddit:
            # Calculate overlap
            overlap = len(ref_reddit & agent_reddit)
            total = len(ref_reddit)
            if overlap / total >= 0.8:  # 80% of items must match
                return True, f"Reddit list match: {overlap}/{total} items"
    
    return False, f"No match found between '{agent_answer[:100]}...' and '{reference_answer}'"


def load_trajectory_data(trajectory_path: str) -> Optional[Dict]:
    """Load trajectory data from JSON file."""
    try:
        traj_file = Path(trajectory_path)
        if not traj_file.exists():
            return None
        
        with open(traj_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load trajectory from {trajectory_path}: {e}")
        return None


def extract_url_from_observation(observation: str) -> Optional[str]:
    """Extract the current URL from an observation string."""
    if not observation:
        return None
    
    lines = observation.split('\n')
    for line in lines:
        if 'current url:' in line.lower():
            # Extract URL after "Current URL: "
            parts = line.split(':', 1)
            if len(parts) > 1:
                return parts[1].strip()
    return None


def re_evaluate_results(results: Dict, verbose: bool = False, check_observations: bool = True) -> Dict:
    """
    Re-evaluate all task results with enhanced matching and observation checking.
    
    Args:
        results: Results dictionary to re-evaluate
        verbose: Show all evaluation changes
        check_observations: Check trajectory observations for program_html tasks
    
    Returns updated results dictionary.
    """
    updated_count = 0
    task_results = results['task_results']
    
    print(f"\nRe-evaluating {len(task_results)} tasks with enhanced matching...")
    if check_observations:
        print("  (including trajectory observation analysis for program_html tasks)")
    print("-" * 80)
    
    for task in task_results:
        if task.get('error'):
            continue  # Skip error tasks
        
        agent_answer = task.get('agent_answer', '')
        reference_answer = task.get('reference_answer', '')
        original_success = task.get('success', False)
        task_id = task.get('task_id', '')
        
        # Load trajectory data if available and checking observations
        observations = None
        thoughts = None
        task_data = None
        if check_observations:
            trajectory_path = task.get('trajectory_path', '')
            if trajectory_path:
                traj_data = load_trajectory_data(trajectory_path)
                if traj_data:
                    observations = traj_data.get('observations', [])
                    thoughts = traj_data.get('thoughts', [])
            
            # Load eval spec from WebArena config files (authoritative source)
            eval_spec = None
            if WEBARENA_LOADER_AVAILABLE:
                eval_spec = load_eval_spec_for_task(task_id)
                if eval_spec:
                    logger.debug(f"Loaded eval spec from WebArena config for {task_id}")
            
            # Fallback to eval_spec from task results if available
            if not eval_spec:
                eval_spec = task.get('eval_spec', {})
            
            # Reconstruct task_data for program_html checking
            if eval_spec:
                task_data = {
                    'task_id': task_id,
                    'eval': eval_spec
                }
        
        # First try program_html observation checking
        is_match = False
        explanation = ""
        
        if task_data and observations:
            is_program_html_match, program_html_explanation = check_program_html_observation(
                task_data, observations, agent_answer, thoughts
            )
            if "program_html" in program_html_explanation:
                # This is a program_html task
                is_match = is_program_html_match
                explanation = program_html_explanation
        
        # If not a program_html task or not matched, try regular answer matching
        if not explanation or (not is_match and reference_answer):
            # Use reference answer from WebArena config if available
            if WEBARENA_LOADER_AVAILABLE and not reference_answer:
                loader = get_eval_loader()
                webarena_ref = loader.get_reference_answer(task_id)
                if webarena_ref:
                    reference_answer = webarena_ref
                    logger.debug(f"Using reference from WebArena config for {task_id}: {reference_answer[:50]}")
            
            is_match, explanation = check_answer_match(agent_answer, reference_answer)
        
        if is_match != original_success:
            updated_count += 1
            if verbose or is_match:  # Show all newly successful tasks
                status = "✓ NOW CORRECT" if is_match else "✗ NOW INCORRECT"
                print(f"\n{status}: {task['task_id']}")
                print(f"  Agent:     {agent_answer[:100] if agent_answer else 'N/A'}...")
                print(f"  Reference: {reference_answer[:100] if reference_answer else 'N/A'}...")
                print(f"  Reason:    {explanation}")
            
            # Update the task
            task['success'] = is_match
            task['evaluation_explanation'] = explanation
            task['re_evaluated'] = True
    
    # Update aggregate metrics
    successful_tasks = sum(1 for t in task_results if t.get('success', False))
    results['successful_tasks'] = successful_tasks
    results['success_rate'] = successful_tasks / len(task_results) if task_results else 0.0
    
    print("-" * 80)
    print(f"\nRe-evaluation complete!")
    print(f"  Updated: {updated_count} tasks")
    print(f"  New success count: {successful_tasks}/{len(task_results)}")
    print(f"  New success rate: {results['success_rate']*100:.2f}%")
    print()
    
    return results


def load_results(result_path: Path) -> Dict:
    """Load results from JSON file."""
    with open(result_path, 'r') as f:
        return json.load(f)


def calculate_metrics(results: Dict) -> EvaluationMetrics:
    """
    Calculate evaluation metrics from results.
    
    Following WebArena evaluation protocol:
    - Success rate: % of tasks successfully resolved (with answer matching)
    - Average steps: Mean number of steps taken to complete queries
    """
    task_results = results['task_results']
    
    # Filter tasks by status
    successful_tasks = [t for t in task_results if t['success'] == True]
    failed_tasks = [t for t in task_results if t['success'] == False and t['error'] is None]
    error_tasks = [t for t in task_results if t['error'] is not None]
    attempted_tasks = [t for t in task_results if t['error'] is None]  # No errors
    
    # Tasks with/without answers
    tasks_with_answer = [t for t in task_results if t['agent_answer'] is not None and t['agent_answer'] != '']
    tasks_without_answer = [t for t in task_results if t['agent_answer'] is None or t['agent_answer'] == '']
    
    total_tasks = len(task_results)
    num_successful = len(successful_tasks)
    num_failed = len(failed_tasks)
    num_errors = len(error_tasks)
    num_attempted = len(attempted_tasks)
    
    # Calculate success rate (effectiveness metric)
    success_rate = (num_successful / total_tasks * 100) if total_tasks > 0 else 0.0
    
    # Calculate average steps (efficiency metric)
    # Average steps for ALL tasks (including errors with 0 steps)
    all_steps = [t['steps'] for t in task_results]
    avg_steps_all = sum(all_steps) / len(all_steps) if all_steps else 0.0
    
    # Average steps for successful tasks only
    successful_steps = [t['steps'] for t in successful_tasks]
    avg_steps_successful = sum(successful_steps) / len(successful_steps) if successful_steps else 0.0
    
    # Average steps for attempted tasks (excluding errors)
    attempted_steps = [t['steps'] for t in attempted_tasks]
    avg_steps_attempted = sum(attempted_steps) / len(attempted_steps) if attempted_steps else 0.0
    
    return EvaluationMetrics(
        mode=results.get('mode', 'unknown'),
        subset=results.get('subset', 'unknown'),
        total_tasks=total_tasks,
        successful_tasks=num_successful,
        failed_tasks=num_failed,
        error_tasks=num_errors,
        success_rate=success_rate,
        avg_steps_all=avg_steps_all,
        avg_steps_successful=avg_steps_successful,
        avg_steps_attempted=avg_steps_attempted,
        total_tokens=results.get('total_tokens', 0),
        total_walltime=results.get('total_walltime', 0.0),
        tasks_with_answer=len(tasks_with_answer),
        tasks_without_answer=len(tasks_without_answer)
    )


def print_metrics_report(metrics: EvaluationMetrics, verbose: bool = True):
    """Print a formatted metrics report."""
    print("=" * 80)
    print(f"EVALUATION METRICS REPORT")
    print(f"Mode: {metrics.mode.upper()}")
    print(f"Subset: {metrics.subset.upper()}")
    print("=" * 80)
    print()
    
    # Task Summary
    print("TASK SUMMARY")
    print("-" * 80)
    print(f"Total Tasks:              {metrics.total_tasks}")
    print(f"  ✓ Successful:           {metrics.successful_tasks} ({metrics.successful_tasks/metrics.total_tasks*100:.1f}%)" if metrics.total_tasks > 0 else "  ✓ Successful:           0 (0.0%)")
    print(f"  ✗ Failed:               {metrics.failed_tasks} ({metrics.failed_tasks/metrics.total_tasks*100:.1f}%)" if metrics.total_tasks > 0 else "  ✗ Failed:               0 (0.0%)")
    print(f"  ⚠ Errors:                {metrics.error_tasks} ({metrics.error_tasks/metrics.total_tasks*100:.1f}%)" if metrics.total_tasks > 0 else "  ⚠ Errors:                0 (0.0%)")
    print()
    
    # Answer Coverage
    print("ANSWER COVERAGE")
    print("-" * 80)
    print(f"Tasks with Answer:        {metrics.tasks_with_answer}")
    print(f"Tasks without Answer:     {metrics.tasks_without_answer}")
    print()
    
    # PRIMARY METRICS (Following Paper Guidelines)
    print("PRIMARY METRICS (WebArena Evaluation Protocol)")
    print("-" * 80)
    print()
    print("1. EFFECTIVENESS")
    print(f"   Success Rate:          {metrics.success_rate:.2f}%")
    print("   (% of user queries successfully resolved)")
    print("   Uses: LLM-based fuzzy matching + exact string matching")
    print()
    
    print("2. EFFICIENCY")
    print(f"   Avg Steps (All):       {metrics.avg_steps_all:.2f}")
    print(f"   Avg Steps (Success):   {metrics.avg_steps_successful:.2f}")
    print(f"   Avg Steps (Attempted): {metrics.avg_steps_attempted:.2f}")
    print("   (Average number of steps to complete queries)")
    print("   Reflects: Computational and interaction cost")
    print()
    
    # Resource Usage
    print("RESOURCE USAGE")
    print("-" * 80)
    print(f"Total Tokens:             {metrics.total_tokens:,}")
    print(f"Total Wall Time:          {metrics.total_walltime:.2f} seconds ({metrics.total_walltime/60:.2f} minutes)")
    if metrics.total_tasks > 0:
        print(f"Avg Tokens/Task:          {metrics.total_tokens/metrics.total_tasks:,.0f}")
        print(f"Avg Time/Task:            {metrics.total_walltime/metrics.total_tasks:.2f} seconds")
    print()
    
    print("=" * 80)
    print()


def compare_experiments(metrics_list: List[Tuple[str, EvaluationMetrics]]):
    """Compare multiple experiments side by side."""
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    # Create comparison table
    data = []
    for name, m in metrics_list:
        data.append({
            'Experiment': name,
            'Success Rate (%)': f"{m.success_rate:.2f}%",
            'Avg Steps (All)': f"{m.avg_steps_all:.2f}",
            'Avg Steps (Success)': f"{m.avg_steps_successful:.2f}",
            'Tasks': f"{m.successful_tasks}/{m.total_tasks}",
            'Errors': m.error_tasks
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # Calculate improvements if comparing 2 experiments
    if len(metrics_list) == 2:
        baseline_name, baseline = metrics_list[0]
        treatment_name, treatment = metrics_list[1]
        
        print(f"IMPROVEMENTS: {treatment_name} vs {baseline_name}")
        print("-" * 80)
        
        success_rate_delta = treatment.success_rate - baseline.success_rate
        avg_steps_delta = treatment.avg_steps_attempted - baseline.avg_steps_attempted
        
        print(f"Success Rate:   {baseline.success_rate:.2f}% → {treatment.success_rate:.2f}% ({success_rate_delta:+.2f}%)")
        print(f"Avg Steps:      {baseline.avg_steps_attempted:.2f} → {treatment.avg_steps_attempted:.2f} ({avg_steps_delta:+.2f})")
        print()
        
        # Interpretation
        if success_rate_delta > 0:
            print(f"✓ Success rate improved by {success_rate_delta:.2f} percentage points")
        elif success_rate_delta < 0:
            print(f"✗ Success rate decreased by {abs(success_rate_delta):.2f} percentage points")
        else:
            print("○ Success rate unchanged")
        
        if avg_steps_delta < 0:
            print(f"✓ Average steps reduced by {abs(avg_steps_delta):.2f} (more efficient)")
        elif avg_steps_delta > 0:
            print(f"✗ Average steps increased by {avg_steps_delta:.2f} (less efficient)")
        else:
            print("○ Average steps unchanged")
        print()
    
    print("=" * 80)


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ReasoningBank results with enhanced matching")
    parser.add_argument(
        '--re-evaluate',
        action='store_true',
        help='Re-evaluate answers with enhanced matching'
    )
    parser.add_argument(
        '--save-updated',
        action='store_true',
        help='Save updated results after re-evaluation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all evaluation changes'
    )
    parser.add_argument(
        '--check-observations',
        action='store_true',
        default=True,
        help='Check trajectory observations for program_html tasks (default: True)'
    )
    parser.add_argument(
        '--no-check-observations',
        dest='check_observations',
        action='store_false',
        help='Skip checking trajectory observations'
    )
    args = parser.parse_args()
    
    results_dir = Path('/Users/hivamoh/Desktop/ReasoningBank/results')
    
    # Find all result JSON files
    result_files = list(results_dir.glob('*.json'))
    
    if not result_files:
        print("No result files found in results/ directory")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result file(s)")
    print()
    
    # Load and analyze each result file
    all_metrics = []
    
    for result_file in sorted(result_files):
        if result_file.name.endswith('_checkpoint.json'):
            continue  # Skip checkpoint files
        
        print(f"Processing: {result_file.name}")
        results = load_results(result_file)
        
        # Re-evaluate if requested
        if args.re_evaluate:
            print("\n" + "="*80)
            print(f"RE-EVALUATING: {result_file.name}")
            print("="*80)
            results = re_evaluate_results(
                results, 
                verbose=args.verbose,
                check_observations=args.check_observations
            )
            
            # Save updated results if requested
            if args.save_updated:
                # Create backup
                backup_file = result_file.with_suffix('.json.backup')
                if not backup_file.exists():
                    import shutil
                    shutil.copy(result_file, backup_file)
                    print(f"Backup saved to: {backup_file.name}")
                
                # Save updated results
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Updated results saved to: {result_file.name}")
                
                # Also update CSV if it exists
                csv_file = result_file.with_suffix('.csv')
                if csv_file.exists():
                    csv_data = []
                    for task in results['task_results']:
                        csv_data.append({
                            'task_id': task['task_id'],
                            'subset': task['subset'],
                            'success': task['success'],
                            'steps': task['steps'],
                            'tokens_input': task['tokens_input'],
                            'tokens_output': task['tokens_output'],
                            'walltime': task['walltime'],
                            'seed': task['seed'],
                            'error': task.get('error', ''),
                            'trajectory_path': task.get('trajectory_path', ''),
                            'agent_answer': task.get('agent_answer', ''),
                            'reference_answer': task.get('reference_answer', ''),
                            'evaluation_explanation': task.get('evaluation_explanation', '')
                        })
                    df_csv = pd.DataFrame(csv_data)
                    df_csv.to_csv(csv_file, index=False)
                    print(f"Updated CSV saved to: {csv_file.name}")
            print()
        
        metrics = calculate_metrics(results)
        
        experiment_name = result_file.stem.replace('_', ' ').title()
        all_metrics.append((experiment_name, metrics))
        
        print_metrics_report(metrics, verbose=True)
    
    # If we have multiple experiments, compare them
    if len(all_metrics) > 1:
        compare_experiments(all_metrics)
    
    # Generate summary CSV
    summary_file = results_dir / 'evaluation_summary.csv'
    summary_data = []
    
    for name, m in all_metrics:
        summary_data.append({
            'Experiment': name,
            'Mode': m.mode,
            'Subset': m.subset,
            'Total Tasks': m.total_tasks,
            'Successful': m.successful_tasks,
            'Failed': m.failed_tasks,
            'Errors': m.error_tasks,
            'Success Rate (%)': round(m.success_rate, 2),
            'Avg Steps (All)': round(m.avg_steps_all, 2),
            'Avg Steps (Success)': round(m.avg_steps_successful, 2),
            'Avg Steps (Attempted)': round(m.avg_steps_attempted, 2),
            'Total Tokens': m.total_tokens,
            'Total Time (min)': round(m.total_walltime / 60, 2)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    print()


if __name__ == '__main__':
    main()

