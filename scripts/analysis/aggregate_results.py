#!/usr/bin/env python3
"""
Aggregate results from multiple evaluation runs and generate Table 1.

Usage:
    python aggregate_results.py --results_dir results --output table1.csv
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from loguru import logger


def load_results(results_dir: Path, mode: str, subsets: List[str]) -> Dict[str, dict]:
    """Load evaluation results for a mode across subsets."""
    
    results = {}
    
    for subset in subsets:
        result_file = results_dir / f"{mode}_{subset}.json"
        
        if not result_file.exists():
            logger.warning(f"Result file not found: {result_file}")
            continue
        
        with open(result_file, "r") as f:
            data = json.load(f)
            results[subset] = data
    
    return results


def compute_overall_metrics(subset_results: Dict[str, dict]) -> dict:
    """Compute overall metrics across all subsets."""
    
    total_tasks = 0
    total_successful = 0
    all_steps = []
    total_tokens = 0
    total_walltime = 0.0
    
    for subset, data in subset_results.items():
        total_tasks += data["total_tasks"]
        total_successful += data["successful_tasks"]
        
        # Get steps from individual task results
        for task in data["task_results"]:
            if task["success"] and task["steps"] > 0:
                all_steps.append(task["steps"])
        
        total_tokens += data["total_tokens"]
        total_walltime += data["total_walltime"]
    
    success_rate = total_successful / total_tasks if total_tasks > 0 else 0.0
    avg_steps = sum(all_steps) / len(all_steps) if all_steps else 0.0
    
    return {
        "subset": "Overall",
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "total_tasks": total_tasks,
        "successful_tasks": total_successful,
        "total_tokens": total_tokens,
        "total_walltime": total_walltime
    }


def generate_comparison_table(
    baseline_results: Dict[str, dict],
    rb_results: Dict[str, dict],
    subsets: List[str]
) -> pd.DataFrame:
    """Generate Table 1 comparison."""
    
    rows = []
    
    # Per-subset results
    for subset in subsets:
        baseline = baseline_results.get(subset, {})
        rb = rb_results.get(subset, {})
        
        baseline_sr = baseline.get("success_rate", 0.0)
        baseline_steps = baseline.get("avg_steps", 0.0)
        
        rb_sr = rb.get("success_rate", 0.0)
        rb_steps = rb.get("avg_steps", 0.0)
        
        delta_sr = rb_sr - baseline_sr
        delta_steps = rb_steps - baseline_steps
        
        rows.append({
            "Subset": subset.capitalize(),
            "No-Memory SR": f"{baseline_sr:.3f}",
            "No-Memory Steps": f"{baseline_steps:.1f}",
            "ReasoningBank SR": f"{rb_sr:.3f}",
            "ReasoningBank Steps": f"{rb_steps:.1f}",
            "ΔSR": f"{delta_sr:+.3f}",
            "ΔSteps": f"{delta_steps:+.1f}",
            "Total Tasks": baseline.get("total_tasks", 0)
        })
    
    # Overall results
    baseline_overall = compute_overall_metrics(baseline_results)
    rb_overall = compute_overall_metrics(rb_results)
    
    delta_sr_overall = rb_overall["success_rate"] - baseline_overall["success_rate"]
    delta_steps_overall = rb_overall["avg_steps"] - baseline_overall["avg_steps"]
    
    rows.append({
        "Subset": "OVERALL",
        "No-Memory SR": f"{baseline_overall['success_rate']:.3f}",
        "No-Memory Steps": f"{baseline_overall['avg_steps']:.1f}",
        "ReasoningBank SR": f"{rb_overall['success_rate']:.3f}",
        "ReasoningBank Steps": f"{rb_overall['avg_steps']:.1f}",
        "ΔSR": f"{delta_sr_overall:+.3f}",
        "ΔSteps": f"{delta_steps_overall:+.1f}",
        "Total Tasks": baseline_overall["total_tasks"]
    })
    
    df = pd.DataFrame(rows)
    return df


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    baseline_results: Dict[str, dict],
    rb_results: Dict[str, dict],
    output_dir: Path
):
    """Generate markdown report with analysis."""
    
    report_lines = [
        "# ReasoningBank Table 1 Reproduction Results",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Comparison Table",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Analysis",
        "",
    ]
    
    # Extract overall row
    overall = comparison_df[comparison_df["Subset"] == "OVERALL"].iloc[0]
    
    delta_sr = float(overall["ΔSR"])
    delta_steps = float(overall["ΔSteps"])
    
    report_lines.extend([
        f"### Overall Performance",
        "",
        f"- **Success Rate Improvement:** {delta_sr:+.3f} ({delta_sr*100:+.1f}%)",
        f"- **Steps Change:** {delta_steps:+.1f}",
        "",
        "### Success Criteria Assessment",
        "",
        f"- Primary Metric (ΔSR > 0): {'✅ PASS' if delta_sr > 0 else '❌ FAIL'}",
        f"- Secondary Metric (ΔSteps < 0): {'✅ PASS' if delta_steps < 0 else '❌ FAIL'}",
        "",
    ])
    
    # Subset-level analysis
    report_lines.extend([
        "### Per-Subset Analysis",
        "",
    ])
    
    positive_sr_count = 0
    negative_steps_count = 0
    
    for _, row in comparison_df.iterrows():
        if row["Subset"] == "OVERALL":
            continue
        
        subset = row["Subset"]
        d_sr = float(row["ΔSR"])
        d_steps = float(row["ΔSteps"])
        
        if d_sr > 0:
            positive_sr_count += 1
        if d_steps < 0:
            negative_steps_count += 1
        
        sr_icon = "✅" if d_sr > 0 else "❌"
        steps_icon = "✅" if d_steps < 0 else "❌"
        
        report_lines.append(
            f"- **{subset}**: SR {sr_icon} ({d_sr:+.3f}), Steps {steps_icon} ({d_steps:+.1f})"
        )
    
    total_subsets = len(comparison_df) - 1  # Exclude overall
    
    report_lines.extend([
        "",
        f"**Subsets with improved SR:** {positive_sr_count}/{total_subsets}",
        f"**Subsets with reduced steps:** {negative_steps_count}/{total_subsets}",
        "",
    ])
    
    # Memory bank stats (if available)
    if rb_results:
        report_lines.extend([
            "## Memory Bank Statistics",
            "",
        ])
        
        # Try to load memory bank stats
        bank_path = Path("memory_bank") / "memories.jsonl"
        if bank_path.exists():
            with open(bank_path, "r") as f:
                num_memories = sum(1 for _ in f)
            
            report_lines.append(f"- Total memories extracted: {num_memories}")
        
        report_lines.append("")
    
    # Comparison to paper
    report_lines.extend([
        "## Comparison to Paper Results",
        "",
        "**Expected from Paper (WebArena):**",
        "- Gemini 2.5 Flash: +8.3% SR improvement",
        "- Gemini 2.5 Pro: +7.2% SR improvement",
        "",
        f"**Our Results:** {delta_sr*100:+.1f}% SR improvement",
        "",
        "**Note:** Direct comparison may not be exact due to:",
        "- Different LLM backbone used",
        "- Potential differences in implementation details",
        "- Different random seeds",
        "- Simplified mock environment (if applicable)",
        "",
    ])
    
    # Save report
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing evaluation results"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="table1_comparison.csv",
        help="Output CSV file for Table 1"
    )
    
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["shopping", "admin", "gitlab", "reddit", "multi"],
        help="Subsets to include"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    logger.info(f"Loading results from {results_dir}")
    
    # Load results for both modes
    baseline_results = load_results(results_dir, "no_memory", args.subsets)
    rb_results = load_results(results_dir, "reasoningbank", args.subsets)
    
    if not baseline_results or not rb_results:
        logger.error("Missing results for one or both modes")
        logger.info("Make sure to run both modes first:")
        logger.info("  python run_eval.py --mode no_memory --subset all")
        logger.info("  python run_eval.py --mode reasoningbank --subset all")
        return
    
    logger.info(f"Loaded results for {len(baseline_results)} subsets")
    
    # Generate comparison table
    comparison_df = generate_comparison_table(baseline_results, rb_results, args.subsets)
    
    # Save table
    output_path = results_dir / args.output
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")
    
    # Print table
    print("\n" + "="*80)
    print("TABLE 1: No-Memory vs ReasoningBank Comparison")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Generate markdown report
    generate_markdown_report(comparison_df, baseline_results, rb_results, results_dir)
    
    logger.info("Aggregation complete!")


if __name__ == "__main__":
    main()
