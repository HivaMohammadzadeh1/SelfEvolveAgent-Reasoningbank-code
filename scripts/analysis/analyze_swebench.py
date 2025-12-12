#!/usr/bin/env python3
"""
Analyze SWE-bench results by comparing final_answer to reference_answer.
Ignores the success field and determines success based on similarity.
"""

import json
import glob
import os
from difflib import SequenceMatcher
import re


def extract_file_from_diff(diff_text):
    """Extract the file path being modified from a diff."""
    if not diff_text:
        return None
    # Look for lines like "diff --git a/path/to/file.py b/path/to/file.py"
    match = re.search(r'diff --git a/(.*?) b/', diff_text)
    if match:
        return match.group(1)
    # Or lines like "--- a/path/to/file.py"
    match = re.search(r'--- a/(.*?)$', diff_text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def normalize_diff(diff_text):
    """Normalize diff text for comparison."""
    if not diff_text:
        return ""
    # Remove leading/trailing whitespace
    diff_text = diff_text.strip()
    # Normalize line endings
    diff_text = diff_text.replace('\r\n', '\n')
    return diff_text


def are_diffs_similar(final_answer, reference_answer, threshold=0.5):
    """
    Check if final_answer and reference_answer are similar.

    Args:
        final_answer: The predicted diff/patch
        reference_answer: The ground truth diff/patch
        threshold: Similarity threshold (0-1). Lower means more lenient.

    Returns:
        bool: True if diffs are considered similar
    """
    # If final_answer is None or empty, not similar
    if not final_answer or not final_answer.strip():
        return False

    # If reference_answer is None or empty, can't compare
    if not reference_answer or not reference_answer.strip():
        return False

    # Normalize both diffs
    final_norm = normalize_diff(final_answer)
    ref_norm = normalize_diff(reference_answer)

    # Check if they modify the same file
    final_file = extract_file_from_diff(final_norm)
    ref_file = extract_file_from_diff(ref_norm)

    if final_file != ref_file:
        # Different files being modified - not similar
        return False

    # Use SequenceMatcher to compare similarity
    similarity = SequenceMatcher(None, final_norm, ref_norm).ratio()

    # Also check if key parts of the diff are present
    # Extract the actual changes (lines starting with + or -)
    final_changes = [line for line in final_norm.split('\n')
                     if line.startswith('+') or line.startswith('-')]
    ref_changes = [line for line in ref_norm.split('\n')
                   if line.startswith('+') or line.startswith('-')]

    if final_changes and ref_changes:
        changes_similarity = SequenceMatcher(None,
                                            '\n'.join(final_changes),
                                            '\n'.join(ref_changes)).ratio()
        # Use the higher of the two similarities
        similarity = max(similarity, changes_similarity)

    return similarity >= threshold


def analyze_results(directory_path, threshold=0.5):
    """
    Analyze all JSON files in the directory.

    Args:
        directory_path: Path to directory containing JSON files
        threshold: Similarity threshold for considering answers similar

    Returns:
        dict: Statistics about the results
    """
    json_files = glob.glob(os.path.join(directory_path, '*.json'))

    total_tasks = 0
    successful_tasks = 0
    similar_answers = []
    dissimilar_answers = []
    no_final_answer = 0

    # Track steps
    total_steps = 0
    successful_steps = 0
    dissimilar_steps = 0
    no_answer_steps = 0

    successful_task_details = []
    dissimilar_task_details = []
    no_answer_task_details = []

    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {json_file}")
                continue

        task_id = data.get('task_id', os.path.basename(json_file))
        final_answer = data.get('final_answer')
        reference_answer = data.get('reference_answer')
        original_success = data.get('success', False)
        steps = data.get('steps', 0)

        total_tasks += 1
        total_steps += steps

        if final_answer is None or not final_answer.strip():
            no_final_answer += 1
            no_answer_steps += steps
            dissimilar_answers.append(task_id)
            no_answer_task_details.append({'task_id': task_id, 'steps': steps})
            continue

        # Check similarity
        is_similar = are_diffs_similar(final_answer, reference_answer, threshold)

        if is_similar:
            successful_tasks += 1
            successful_steps += steps
            similar_answers.append(task_id)
            successful_task_details.append({'task_id': task_id, 'steps': steps})
        else:
            dissimilar_steps += steps
            dissimilar_answers.append(task_id)
            dissimilar_task_details.append({'task_id': task_id, 'steps': steps})

    # Calculate statistics
    success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
    avg_steps_all = total_steps / total_tasks if total_tasks > 0 else 0
    avg_steps_successful = successful_steps / successful_tasks if successful_tasks > 0 else 0
    avg_steps_dissimilar = dissimilar_steps / (total_tasks - successful_tasks - no_final_answer) if (total_tasks - successful_tasks - no_final_answer) > 0 else 0
    avg_steps_no_answer = no_answer_steps / no_final_answer if no_final_answer > 0 else 0

    stats = {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'success_rate': success_rate,
        'no_final_answer': no_final_answer,
        'similar_answers': similar_answers,
        'dissimilar_answers': dissimilar_answers,
        'total_steps': total_steps,
        'avg_steps_all': avg_steps_all,
        'avg_steps_successful': avg_steps_successful,
        'avg_steps_dissimilar': avg_steps_dissimilar,
        'avg_steps_no_answer': avg_steps_no_answer,
        'successful_task_details': successful_task_details,
        'dissimilar_task_details': dissimilar_task_details,
        'no_answer_task_details': no_answer_task_details
    }

    return stats


def main():
    directory_path = '/Users/hivamoh/Desktop/ReasoningBank/logs/reasoningbank/reddit'

    print("Analyzing SWE-bench results...")
    print(f"Directory: {directory_path}")
    print("=" * 80)

    # Analyze with different thresholds
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n### Threshold: {threshold} ###")
        stats = analyze_results(directory_path, threshold=threshold)

        print(f"Total tasks: {stats['total_tasks']}")
        print(f"Successful tasks: {stats['successful_tasks']}")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        print(f"Tasks with no final_answer: {stats['no_final_answer']}")
        print(f"Tasks with similar answers: {len(stats['similar_answers'])}")
        print(f"Tasks with dissimilar answers: {len(stats['dissimilar_answers'])}")
        print(f"\nStep Statistics:")
        print(f"  Average steps (all tasks): {stats['avg_steps_all']:.2f}")
        print(f"  Average steps (successful): {stats['avg_steps_successful']:.2f}")
        print(f"  Average steps (dissimilar): {stats['avg_steps_dissimilar']:.2f}")
        print(f"  Average steps (no answer): {stats['avg_steps_no_answer']:.2f}")

    print("\n" + "=" * 80)
    print("\nDetailed results (using threshold 0.5):")
    stats = analyze_results(directory_path, threshold=0.5)

    if stats['similar_answers']:
        print(f"\n✓ Similar answers ({len(stats['similar_answers'])}):")
        for task_id in stats['similar_answers'][:10]:  # Show first 10
            print(f"  - {task_id}")
        if len(stats['similar_answers']) > 10:
            print(f"  ... and {len(stats['similar_answers']) - 10} more")

    # Save detailed results to file
    output_file = '/Users/hivamoh/Desktop/ReasoningBank/logs/swebench_no_memory/analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
