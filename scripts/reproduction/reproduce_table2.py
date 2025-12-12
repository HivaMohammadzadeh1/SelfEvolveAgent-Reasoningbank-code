#!/usr/bin/env python3
"""
Script to reproduce Table 2 from ReasoningBank paper.

Table 2: Experiment results of ReasoningBank on SWE-Bench-Verified
dataset for issue-resolving in a given repository.

Usage:
    # Run all experiments for one model
    python reproduce_table2.py --model gemini-2.5-flash

    # Run specific mode only
    python reproduce_table2.py --model gemini-2.5-flash --mode reasoningbank

    # Aggregate existing results
    python reproduce_table2.py --aggregate_only
"""
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
import pandas as pd


def run_experiment(mode: str, model: str, config_file: str, max_tasks: int = None):
    """
    Run a single SWE-bench experiment.

    Args:
        mode: Experiment mode (no_memory, synapse, reasoningbank)
        model: Model name (gemini-2.5-flash, gemini-2.5-pro, etc.)
        config_file: Path to config file
        max_tasks: Optional limit on number of tasks
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {mode} with {model}")
    logger.info(f"{'='*70}\n")

    cmd = [
        "python", "run_swebench.py",
        "--mode", mode,
        "--config", config_file
    ]

    if max_tasks:
        cmd.extend(["--max_tasks", str(max_tasks)])

    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✓ Completed: {mode} with {model}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {mode} with {model}")
        logger.error(f"Error: {e}")
        return False


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load results from a results directory."""
    results_dir = Path(results_dir)
    summary_file = results_dir / "summary.json"

    if not summary_file.exists():
        logger.warning(f"No summary file found at {summary_file}")
        return None

    with open(summary_file, "r") as f:
        return json.load(f)


def aggregate_results(base_results_dir: str, models: List[str]) -> pd.DataFrame:
    """
    Aggregate results across all experiments.

    Args:
        base_results_dir: Base directory for results
        models: List of models to aggregate

    Returns:
        DataFrame with aggregated results
    """
    results = []

    for model in models:
        for mode in ["no_memory", "synapse", "reasoningbank"]:
            results_dir = f"{base_results_dir}/swebench_{mode}"

            data = load_results(results_dir)
            if data:
                results.append({
                    "Model": model,
                    "Method": mode.replace("_", " ").title(),
                    "Resolve Rate": f"{data['success_rate']*100:.1f}",
                    "Step": f"{data['avg_steps']:.1f}"
                })

    if not results:
        logger.error("No results found to aggregate")
        return None

    df = pd.DataFrame(results)
    return df


def format_table2(df: pd.DataFrame) -> str:
    """Format results as Table 2 from the paper."""
    table = "\nTable 2 | Experiment results of ReasoningBank on SWE-Bench-Verified\n"
    table += "=" * 70 + "\n"
    table += f"{'Methods':<25} {'Resolve Rate':<15} {'Step':<10}\n"
    table += "-" * 70 + "\n"

    current_model = None
    for _, row in df.iterrows():
        if current_model != row['Model']:
            if current_model is not None:
                table += "-" * 70 + "\n"
            table += f"\n{row['Model']}\n"
            current_model = row['Model']

        table += f"{row['Method']:<25} {row['Resolve Rate']:<15} {row['Step']:<10}\n"

    table += "=" * 70 + "\n"
    return table


def compare_with_paper(df: pd.DataFrame):
    """Compare results with paper's Table 2."""
    logger.info("\n" + "="*70)
    logger.info("Comparison with Paper Results (Table 2)")
    logger.info("="*70 + "\n")

    # Expected results from paper
    paper_results = {
        "gemini-2.5-flash": {
            "No Memory": {"resolve_rate": 34.2, "steps": 30.3},
            "Synapse": {"resolve_rate": 35.4, "steps": 30.7},
            "Reasoningbank": {"resolve_rate": 38.8, "steps": 27.5}
        },
        "gemini-2.5-pro": {
            "No Memory": {"resolve_rate": 54.0, "steps": 21.1},
            "Synapse": {"resolve_rate": 53.4, "steps": 21.0},
            "Reasoningbank": {"resolve_rate": 57.4, "steps": 19.8}
        }
    }

    for model in df['Model'].unique():
        if model in paper_results:
            logger.info(f"\n{model}:")
            logger.info(f"{'-'*70}")
            logger.info(f"{'Method':<20} {'Our Results':<25} {'Paper Results':<25}")
            logger.info(f"{'-'*70}")

            model_df = df[df['Model'] == model]
            for _, row in model_df.iterrows():
                method = row['Method']
                if method in paper_results[model]:
                    our_rr = row['Resolve Rate']
                    our_steps = row['Step']
                    paper_rr = paper_results[model][method]['resolve_rate']
                    paper_steps = paper_results[model][method]['steps']

                    logger.info(
                        f"{method:<20} "
                        f"RR: {our_rr:>6}%, Steps: {our_steps:>5}   "
                        f"RR: {paper_rr:>5.1f}%, Steps: {paper_steps:>5.1f}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 2 from ReasoningBank paper"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4", "claude-3.5-sonnet"],
        default="gemini-2.5-flash",
        help="Model to use for experiments"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_memory", "synapse", "reasoningbank", "all"],
        default="all",
        help="Which mode to run (default: all)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.swebench.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Limit number of tasks (for testing)"
    )

    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Only aggregate existing results, don't run experiments"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/swebench",
        help="Base directory for results"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("ReasoningBank - Table 2 Reproduction")
    logger.info("SWE-Bench-Verified Evaluation")
    logger.info("="*70)

    # Run experiments if not aggregate_only
    if not args.aggregate_only:
        modes_to_run = ["no_memory", "synapse", "reasoningbank"] if args.mode == "all" else [args.mode]

        logger.info(f"\nRunning experiments for {args.model}")
        logger.info(f"Modes: {modes_to_run}")

        if args.max_tasks:
            logger.warning(f"Limited to {args.max_tasks} tasks (testing mode)")

        for mode in modes_to_run:
            success = run_experiment(
                mode=mode,
                model=args.model,
                config_file=args.config,
                max_tasks=args.max_tasks
            )

            if not success:
                logger.error(f"Failed to complete {mode} experiment")
                logger.error("Continuing with remaining experiments...")

    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("Aggregating Results")
    logger.info("="*70)

    models = [args.model] if not args.aggregate_only else [
        "gemini-2.5-flash", "gemini-2.5-pro"
    ]

    df = aggregate_results(args.results_dir, models)

    if df is not None:
        # Print formatted table
        table = format_table2(df)
        print(table)

        # Save to CSV
        output_file = f"{args.results_dir}/table2_comparison.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to: {output_file}")

        # Compare with paper
        compare_with_paper(df)
    else:
        logger.error("Failed to aggregate results")

    logger.info("\n" + "="*70)
    logger.info("Table 2 reproduction complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
