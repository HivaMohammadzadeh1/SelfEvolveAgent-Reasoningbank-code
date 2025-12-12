#!/usr/bin/env python3
"""
Compare baseline vs optimized results.

Usage:
    python scripts/compare_results.py --subset shopping
    python scripts/compare_results.py --subset all
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    if not filepath.exists():
        return None
    
    with open(filepath) as f:
        return json.load(f)


def print_comparison(baseline: Dict, optimized: Dict, subset: str):
    """Print comparison between baseline and optimized results."""
    
    print("=" * 70)
    print(f"RESULTS COMPARISON - {subset.upper()}")
    print("=" * 70)
    print()
    
    # Extract metrics
    baseline_sr = baseline.get("success_rate", 0.0)
    optimized_sr = optimized.get("success_rate", 0.0)
    
    baseline_steps = baseline.get("avg_steps", 0.0)
    optimized_steps = optimized.get("avg_steps", 0.0)
    
    baseline_tokens = baseline.get("total_tokens", 0)
    optimized_tokens = optimized.get("total_tokens", 0)
    
    baseline_time = baseline.get("total_walltime", 0.0)
    optimized_time = optimized.get("total_walltime", 0.0)
    
    # Improvements
    sr_improvement = optimized_sr - baseline_sr
    steps_improvement = baseline_steps - optimized_steps  # Lower is better
    tokens_improvement = baseline_tokens - optimized_tokens  # Lower is better
    time_improvement = baseline_time - optimized_time  # Lower is better
    
    # Print table
    print(f"{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 70)
    
    print(f"{'Success Rate':<25} {baseline_sr:>14.2%} {optimized_sr:>14.2%} {sr_improvement:>+14.2%}")
    print(f"{'Avg Steps':<25} {baseline_steps:>14.1f} {optimized_steps:>14.1f} {steps_improvement:>+14.1f}")
    print(f"{'Total Tasks':<25} {baseline['total_tasks']:>14} {optimized['total_tasks']:>14} {'':<15}")
    print(f"{'Successful Tasks':<25} {baseline['successful_tasks']:>14} {optimized['successful_tasks']:>14} {'':<15}")
    print(f"{'Total Tokens':<25} {baseline_tokens:>14,} {optimized_tokens:>14,} {tokens_improvement:>+14,}")
    print(f"{'Total Time (s)':<25} {baseline_time:>14.1f} {optimized_time:>14.1f} {time_improvement:>+14.1f}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Overall assessment
    if sr_improvement > 0:
        print(f"✅ Success rate IMPROVED by {sr_improvement:.2%}")
    elif sr_improvement < 0:
        print(f"❌ Success rate DECREASED by {sr_improvement:.2%}")
    else:
        print(f"➖ Success rate UNCHANGED")
    
    if steps_improvement > 0:
        print(f"✅ Efficiency IMPROVED - {steps_improvement:.1f} fewer steps on average")
    elif steps_improvement < 0:
        print(f"❌ Efficiency DECREASED - {abs(steps_improvement):.1f} more steps on average")
    else:
        print(f"➖ Efficiency UNCHANGED")
    
    if tokens_improvement > 0:
        print(f"✅ Token usage REDUCED by {tokens_improvement:,} tokens")
    elif tokens_improvement < 0:
        print(f"❌ Token usage INCREASED by {abs(tokens_improvement):,} tokens")
    else:
        print(f"➖ Token usage UNCHANGED")
    
    print()
    
    # ROI calculation (rough estimate)
    if baseline_sr > 0:
        relative_improvement = (optimized_sr / baseline_sr - 1) * 100
        print(f"📊 Relative improvement: {relative_improvement:+.1f}%")
    
    print()


def compare_task_level(baseline: Dict, optimized: Dict, subset: str):
    """Compare individual task results."""
    
    baseline_tasks = {t["task_id"]: t for t in baseline.get("task_results", [])}
    optimized_tasks = {t["task_id"]: t for t in optimized.get("task_results", [])}
    
    # Find common tasks
    common_task_ids = set(baseline_tasks.keys()) & set(optimized_tasks.keys())
    
    if not common_task_ids:
        print("⚠️  No common tasks found for detailed comparison")
        return
    
    print("=" * 70)
    print(f"TASK-LEVEL COMPARISON - {len(common_task_ids)} tasks")
    print("=" * 70)
    print()
    
    # Tasks that improved
    improved = []
    regressed = []
    unchanged = []
    
    for task_id in common_task_ids:
        baseline_success = baseline_tasks[task_id]["success"]
        optimized_success = optimized_tasks[task_id]["success"]
        
        if not baseline_success and optimized_success:
            improved.append(task_id)
        elif baseline_success and not optimized_success:
            regressed.append(task_id)
        else:
            unchanged.append(task_id)
    
    print(f"✅ Improved (failed → success): {len(improved)} tasks")
    if improved[:5]:
        print(f"   Examples: {', '.join(improved[:5])}")
    
    print(f"❌ Regressed (success → failed): {len(regressed)} tasks")
    if regressed[:5]:
        print(f"   Examples: {', '.join(regressed[:5])}")
    
    print(f"➖ Unchanged: {len(unchanged)} tasks")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs optimized results"
    )
    parser.add_argument(
        "--subset",
        default="shopping",
        help="Subset to compare (shopping, admin, gitlab, reddit, multi, all)"
    )
    parser.add_argument(
        "--baseline_dir",
        default="results",
        help="Directory containing baseline results"
    )
    parser.add_argument(
        "--optimized_dir",
        default="results/optimized",
        help="Directory containing optimized results"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show task-level comparison"
    )
    
    args = parser.parse_args()
    
    # Load results
    baseline_file = Path(args.baseline_dir) / f"no_memory_{args.subset}.json"
    optimized_file = Path(args.optimized_dir) / f"no_memory_{args.subset}.json"
    
    # Try alternative naming
    if not baseline_file.exists():
        baseline_file = Path(args.baseline_dir) / f"reasoningbank_{args.subset}.json"
    
    if not optimized_file.exists():
        optimized_file = Path(args.optimized_dir) / f"reasoningbank_{args.subset}.json"
    
    baseline = load_results(baseline_file)
    optimized = load_results(optimized_file)
    
    if baseline is None:
        print(f"❌ Baseline results not found: {baseline_file}")
        print(f"   Run: python run_eval.py --subset {args.subset}")
        return
    
    if optimized is None:
        print(f"❌ Optimized results not found: {optimized_file}")
        print(f"   Run: python run_eval_optimized.py --subset {args.subset}")
        return
    
    # Print comparison
    print_comparison(baseline, optimized, args.subset)
    
    if args.detailed:
        compare_task_level(baseline, optimized, args.subset)


if __name__ == "__main__":
    main()

