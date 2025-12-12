#!/usr/bin/env python3
"""
Comprehensive Ablation Studies for ReasoningBank

This script runs ablation studies to understand:
1. Effect of different retrieve_k values (number of memories retrieved)
2. Effect of memory from successes only vs successes+failures
3. Effect of different memory extraction temperatures
4. Effect of different embedding models
5. Effect of memory bank size over time
"""

import argparse
import subprocess
import yaml
import json
from pathlib import Path
from loguru import logger
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AblationExperiment:
    """Base class for ablation experiments."""

    def __init__(self, base_config_path: str, output_dir: str):
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        with open(self.base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

    def modify_config(self, changes: Dict) -> Path:
        """Create a modified config file."""
        config = self.base_config.copy()

        # Apply changes (supports nested keys with dots)
        for key, value in changes.items():
            keys = key.split(".")
            current = config
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value

        # Save to temporary config file
        config_path = self.output_dir / f"config_{hash(str(changes))}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def run_experiment(
        self,
        config_path: Path,
        mode: str,
        subset: str,
        exp_name: str
    ) -> Dict:
        """Run a single experiment."""
        logger.info(f"Running experiment: {exp_name}")

        # Run evaluation
        cmd = [
            "python", "run_eval.py",
            "--mode", mode,
            "--subset", subset,
            "--config", str(config_path)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✓ Completed {exp_name}")

            # Load results
            result_file = Path(f"results/{mode}_{subset}.json")
            if result_file.exists():
                with open(result_file, "r") as f:
                    return json.load(f)
            return {}

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed {exp_name}")
            logger.error(f"Error: {e}")
            return {}


class RetrieveKAblation(AblationExperiment):
    """Ablation study for different retrieve_k values."""

    def run(self, subset: str = "shopping", k_values: List[int] = None) -> pd.DataFrame:
        """
        Test different values of retrieve_k (number of memories retrieved).

        Args:
            subset: Subset to test on
            k_values: List of k values to test (default: [1, 2, 3, 5, 10])

        Returns:
            DataFrame with results for each k value
        """
        if k_values is None:
            k_values = [0, 1, 2, 3, 5, 10]

        logger.info("Running retrieve_k ablation study")
        results = []

        for k in k_values:
            logger.info(f"Testing retrieve_k = {k}")

            # Modify config
            config_path = self.modify_config({"memory.retrieve_k": k})

            # Run experiment
            result = self.run_experiment(
                config_path, "reasoningbank", subset,
                f"retrieve_k_{k}"
            )

            if result:
                results.append({
                    "retrieve_k": k,
                    "success_rate": result.get("success_rate", 0),
                    "avg_steps": result.get("avg_steps", 0),
                    "total_tasks": result.get("total_tasks", 0)
                })

        df = pd.DataFrame(results)

        # Save results
        output_file = self.output_dir / f"ablation_retrieve_k_{subset}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")

        return df


class MemorySourceAblation(AblationExperiment):
    """Ablation study for memory from different sources."""

    def run(self, subset: str = "shopping") -> pd.DataFrame:
        """
        Compare:
        1. No memory (baseline)
        2. Success-only memories
        3. Failure-only memories
        4. Both success and failure memories (full ReasoningBank)

        Returns:
            DataFrame comparing all configurations
        """
        logger.info("Running memory source ablation study")
        results = []

        configs = [
            ("no_memory", {}),
            ("success_only", {"memory.use_failures": False}),
            ("failures_only", {"memory.use_successes": False}),
            ("both", {"memory.use_successes": True, "memory.use_failures": True})
        ]

        for config_name, changes in configs:
            logger.info(f"Testing configuration: {config_name}")

            # Determine mode
            mode = "no_memory" if config_name == "no_memory" else "reasoningbank"

            if mode == "reasoningbank":
                config_path = self.modify_config(changes)
            else:
                config_path = self.base_config_path

            # Run experiment
            result = self.run_experiment(
                config_path, mode, subset,
                f"memory_source_{config_name}"
            )

            if result:
                results.append({
                    "configuration": config_name,
                    "success_rate": result.get("success_rate", 0),
                    "avg_steps": result.get("avg_steps", 0),
                    "total_tasks": result.get("total_tasks", 0)
                })

        df = pd.DataFrame(results)

        # Save results
        output_file = self.output_dir / f"ablation_memory_source_{subset}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")

        return df


class TemperatureAblation(AblationExperiment):
    """Ablation study for memory extraction temperature."""

    def run(self, subset: str = "shopping", temperatures: List[float] = None) -> pd.DataFrame:
        """
        Test different extraction temperatures.

        Args:
            subset: Subset to test on
            temperatures: List of temperature values (default: [0.0, 0.5, 1.0, 1.5])

        Returns:
            DataFrame with results for each temperature
        """
        if temperatures is None:
            temperatures = [0.0, 0.5, 1.0, 1.5]

        logger.info("Running extraction temperature ablation study")
        results = []

        for temp in temperatures:
            logger.info(f"Testing temperature = {temp}")

            # Modify config
            config_path = self.modify_config({"llm.extractor_temperature": temp})

            # Run experiment
            result = self.run_experiment(
                config_path, "reasoningbank", subset,
                f"temperature_{temp}"
            )

            if result:
                results.append({
                    "temperature": temp,
                    "success_rate": result.get("success_rate", 0),
                    "avg_steps": result.get("avg_steps", 0),
                    "total_tasks": result.get("total_tasks", 0)
                })

        df = pd.DataFrame(results)

        # Save results
        output_file = self.output_dir / f"ablation_temperature_{subset}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")

        return df


class MemoryQuantityAblation(AblationExperiment):
    """Ablation study for memory bank size."""

    def run(self, subset: str = "shopping", max_items_list: List[int] = None) -> pd.DataFrame:
        """
        Test different max_items_per_trajectory values.

        Args:
            subset: Subset to test on
            max_items_list: List of max items per trajectory (default: [1, 2, 3, 5])

        Returns:
            DataFrame with results for each configuration
        """
        if max_items_list is None:
            max_items_list = [1, 2, 3, 5]

        logger.info("Running memory quantity ablation study")
        results = []

        for max_items in max_items_list:
            logger.info(f"Testing max_items_per_trajectory = {max_items}")

            # Modify config
            config_path = self.modify_config({
                "memory.max_items_per_trajectory": max_items
            })

            # Run experiment
            result = self.run_experiment(
                config_path, "reasoningbank", subset,
                f"max_items_{max_items}"
            )

            if result:
                results.append({
                    "max_items_per_trajectory": max_items,
                    "success_rate": result.get("success_rate", 0),
                    "avg_steps": result.get("avg_steps", 0),
                    "total_tasks": result.get("total_tasks", 0)
                })

        df = pd.DataFrame(results)

        # Save results
        output_file = self.output_dir / f"ablation_memory_quantity_{subset}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to: {output_file}")

        return df


def visualize_ablation_results(results_dir: str, output_dir: str = "ablation_plots"):
    """Generate visualizations for all ablation studies."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # Find all ablation CSV files
    for csv_file in results_path.glob("ablation_*.csv"):
        df = pd.read_csv(csv_file)

        # Determine x-axis column (first non-metric column)
        metric_cols = ["success_rate", "avg_steps", "total_tasks"]
        x_col = next(col for col in df.columns if col not in metric_cols)

        # Create plots for success rate and avg steps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Success rate plot
        ax1.plot(df[x_col], df["success_rate"] * 100, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax1.set_ylabel("Success Rate (%)", fontsize=12)
        ax1.set_title(f"Success Rate vs {x_col.replace('_', ' ').title()}", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Average steps plot
        ax2.plot(df[x_col], df["avg_steps"], marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax2.set_ylabel("Average Steps", fontsize=12)
        ax2.set_title(f"Average Steps vs {x_col.replace('_', ' ').title()}", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / f"{csv_file.stem}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for ReasoningBank")
    parser.add_argument("--config", default="config.yaml", help="Base config file")
    parser.add_argument("--output-dir", default="ablation_results", help="Output directory")
    parser.add_argument("--subset", default="shopping", help="Subset to test on")
    parser.add_argument(
        "--studies",
        nargs="+",
        default=["all"],
        choices=["all", "retrieve_k", "memory_source", "temperature", "memory_quantity"],
        help="Which ablation studies to run"
    )
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which studies to run
    studies_to_run = args.studies
    if "all" in studies_to_run:
        studies_to_run = ["retrieve_k", "memory_source", "temperature", "memory_quantity"]

    logger.info(f"Running ablation studies: {studies_to_run}")

    # Run each study
    if "retrieve_k" in studies_to_run:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY 1: Effect of retrieve_k")
        logger.info("="*80)
        ablation = RetrieveKAblation(args.config, output_dir)
        df = ablation.run(args.subset)
        logger.info("\nResults:")
        logger.info(df.to_string(index=False))

    if "memory_source" in studies_to_run:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY 2: Effect of Memory Source")
        logger.info("="*80)
        ablation = MemorySourceAblation(args.config, output_dir)
        df = ablation.run(args.subset)
        logger.info("\nResults:")
        logger.info(df.to_string(index=False))

    if "temperature" in studies_to_run:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY 3: Effect of Extraction Temperature")
        logger.info("="*80)
        ablation = TemperatureAblation(args.config, output_dir)
        df = ablation.run(args.subset)
        logger.info("\nResults:")
        logger.info(df.to_string(index=False))

    if "memory_quantity" in studies_to_run:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY 4: Effect of Memory Quantity")
        logger.info("="*80)
        ablation = MemoryQuantityAblation(args.config, output_dir)
        df = ablation.run(args.subset)
        logger.info("\nResults:")
        logger.info(df.to_string(index=False))

    # Visualize results
    if args.visualize:
        logger.info("\n" + "="*80)
        logger.info("Generating visualizations...")
        logger.info("="*80)
        visualize_ablation_results(output_dir, "ablation_plots")

    logger.info("\n" + "="*80)
    logger.info("Ablation studies complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
