#!/usr/bin/env python3
"""
Visualization script for SWE-bench comprehensive evaluation.

Generates publication-quality plots for:
- Overall performance comparison
- Difficulty breakdown
- Repository analysis
- Issue type analysis
- Memory analysis
- Efficiency comparison
- Ablation study results
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class SWEBenchVisualizer:
    """Visualizer for SWE-bench evaluation results."""

    def __init__(
        self,
        results_dir: str,
        ablation_dir: Optional[str] = None,
        output_dir: str = "visualizations_swebench"
    ):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory with enhanced results
            ablation_dir: Directory with ablation results (optional)
            output_dir: Output directory for plots
        """
        self.results_dir = Path(results_dir)
        self.ablation_dir = Path(ablation_dir) if ablation_dir else None
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_enhanced_results(self) -> Dict[str, Any]:
        """Load enhanced results for all modes."""
        results = {}

        for mode in ["no_memory", "reasoningbank"]:
            result_file = self.results_dir / f"{mode}_swebench_enhanced.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    results[mode] = json.load(f)
                logger.info(f"Loaded results for {mode}")

        return results

    def plot_overall_comparison(self, results: Dict[str, Any]):
        """Plot overall performance comparison."""
        logger.info("Generating overall comparison plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        modes = []
        resolve_rates = []
        avg_steps = []

        for mode in ["no_memory", "reasoningbank"]:
            if mode in results:
                modes.append(mode.replace("_", " ").title())
                resolve_rates.append(results[mode]["resolve_rate"] * 100)
                avg_steps.append(results[mode]["avg_steps"])

        # Resolve rate comparison
        bars1 = ax1.bar(modes, resolve_rates, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax1.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Resolve Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(resolve_rates) * 1.2)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

        # Average steps comparison
        bars2 = ax2.bar(modes, avg_steps, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax2.set_ylabel('Average Steps', fontsize=12, fontweight='bold')
        ax2.set_title('Average Steps Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(avg_steps) * 1.2)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "overall_comparison_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def plot_difficulty_breakdown(self, results: Dict[str, Any]):
        """Plot performance by difficulty level."""
        logger.info("Generating difficulty breakdown plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        difficulties = ['Easy', 'Medium', 'Hard']
        x = np.arange(len(difficulties))
        width = 0.25

        # Collect data for each mode
        mode_data = {}
        for mode in ["no_memory", "reasoningbank"]:
            if mode in results:
                difficulty_breakdown = results[mode].get("difficulty_breakdown", {})
                mode_data[mode] = {
                    'resolve_rates': [
                        difficulty_breakdown.get("easy", {}).get("resolve_rate", 0) * 100,
                        difficulty_breakdown.get("medium", {}).get("resolve_rate", 0) * 100,
                        difficulty_breakdown.get("hard", {}).get("resolve_rate", 0) * 100
                    ],
                    'avg_steps': [
                        difficulty_breakdown.get("easy", {}).get("avg_steps", 0),
                        difficulty_breakdown.get("medium", {}).get("avg_steps", 0),
                        difficulty_breakdown.get("hard", {}).get("avg_steps", 0)
                    ]
                }

        # Plot resolve rates
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for idx, (mode, color) in enumerate(zip(["no_memory", "reasoningbank"], colors[:2])):
            if mode in mode_data:
                offset = (idx - 1) * width
                ax1.bar(x + offset, mode_data[mode]['resolve_rates'],
                       width, label=mode.replace("_", " ").title(), color=color)

        ax1.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Resolve Rate by Difficulty', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(difficulties)
        ax1.legend()
        ax1.set_ylim(0, 100)

        # Plot average steps
        for idx, (mode, color) in enumerate(zip(["no_memory", "reasoningbank"], colors[:2])):
            if mode in mode_data:
                offset = (idx - 1) * width
                ax2.bar(x + offset, mode_data[mode]['avg_steps'],
                       width, label=mode.replace("_", " ").title(), color=color)

        ax2.set_ylabel('Average Steps', fontsize=12, fontweight='bold')
        ax2.set_title('Average Steps by Difficulty', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(difficulties)
        ax2.legend()

        plt.tight_layout()
        output_path = self.output_dir / "difficulty_breakdown_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def plot_repository_analysis(self, results: Dict[str, Any]):
        """Plot performance by repository (top 10)."""
        logger.info("Generating repository analysis plot...")

        if "reasoningbank" not in results:
            logger.warning("No ReasoningBank results found")
            return

        repo_breakdown = results["reasoningbank"].get("repository_breakdown", {})

        # Get top 10 repositories by count
        top_repos = sorted(repo_breakdown.items(),
                          key=lambda x: x[1]["count"],
                          reverse=True)[:10]

        if not top_repos:
            logger.warning("No repository data found")
            return

        repos = [repo for repo, _ in top_repos]
        counts = [data["count"] for _, data in top_repos]
        resolve_rates = [data["resolve_rate"] * 100 for _, data in top_repos]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Task count by repository
        bars1 = ax1.barh(repos, counts, color='#3498db')
        ax1.set_xlabel('Number of Tasks', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Repositories by Task Count', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()

        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}',
                    ha='left', va='center', fontweight='bold')

        # Resolve rate by repository
        bars2 = ax2.barh(repos, resolve_rates, color='#2ecc71')
        ax2.set_xlabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Resolve Rate by Repository', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)

        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%',
                    ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "repository_analysis_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def plot_issue_type_analysis(self, results: Dict[str, Any]):
        """Plot performance by issue type."""
        logger.info("Generating issue type analysis plot...")

        if "reasoningbank" not in results:
            logger.warning("No ReasoningBank results found")
            return

        type_breakdown = results["reasoningbank"].get("issue_type_breakdown", {})

        if not type_breakdown:
            logger.warning("No issue type data found")
            return

        # Filter out types with 0 count
        types = []
        counts = []
        resolve_rates = []

        for issue_type, data in type_breakdown.items():
            if data["count"] > 0:
                types.append(issue_type.replace("_", " ").title())
                counts.append(data["count"])
                resolve_rates.append(data["resolve_rate"] * 100)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(types))
        width = 0.35

        bars1 = ax.bar(x - width/2, counts, width, label='Count', color='#3498db')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, resolve_rates, width, label='Resolve Rate', color='#2ecc71')

        ax.set_xlabel('Issue Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Task Count', fontsize=12, fontweight='bold', color='#3498db')
        ax2.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold', color='#2ecc71')
        ax.set_title('Performance by Issue Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='#3498db')
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.set_ylim(0, 100)

        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        output_path = self.output_dir / "issue_type_analysis_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def plot_memory_analysis(self, results: Dict[str, Any]):
        """Plot memory metrics analysis."""
        logger.info("Generating memory analysis plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        modes = []
        total_memories = []
        success_ratios = []
        uniqueness_ratios = []

        for mode in ["reasoningbank"]:
            if mode in results:
                memory_metrics = results[mode].get("memory_metrics", {})
                if memory_metrics:
                    modes.append(mode.replace("_", " ").title())
                    quantity = memory_metrics.get("quantity", {})
                    quality = memory_metrics.get("quality", {})

                    total_memories.append(quantity.get("total_memories", 0))
                    success_ratios.append(quantity.get("success_ratio", 0) * 100)
                    uniqueness_ratios.append(quality.get("uniqueness_ratio", 0) * 100)

        if not modes:
            logger.warning("No memory metrics found")
            return

        # Total memories
        axes[0, 0].bar(modes, total_memories, color=['#3498db', '#2ecc71'])
        axes[0, 0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Total Memories Accumulated', fontsize=12, fontweight='bold')

        # Success ratio
        axes[0, 1].bar(modes, success_ratios, color=['#3498db', '#2ecc71'])
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Success Memory Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 100)

        # Uniqueness ratio
        axes[1, 0].bar(modes, uniqueness_ratios, color=['#3498db', '#2ecc71'])
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Memory Uniqueness Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim(0, 100)

        # Comparison: resolve rate vs memory size
        resolve_rates = []
        for mode_key in ["reasoningbank"]:
            if mode_key in results:
                resolve_rates.append(results[mode_key]["resolve_rate"] * 100)

        if len(modes) == len(resolve_rates):
            axes[1, 1].scatter(total_memories, resolve_rates, s=200, c=['#3498db', '#2ecc71'])
            axes[1, 1].set_xlabel('Total Memories', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Memory Size vs Performance', fontsize=12, fontweight='bold')

            for i, mode in enumerate(modes):
                axes[1, 1].annotate(mode, (total_memories[i], resolve_rates[i]),
                                   xytext=(10, 10), textcoords='offset points')

        plt.tight_layout()
        output_path = self.output_dir / "memory_analysis_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def plot_ablation_studies(self):
        """Plot all ablation study results."""
        if not self.ablation_dir or not self.ablation_dir.exists():
            logger.warning("No ablation directory found")
            return

        logger.info("Generating ablation study plots...")

        # Plot retrieve_k ablation
        self._plot_ablation_retrieve_k()

        # Plot memory source ablation
        self._plot_ablation_memory_source()

        # Plot temperature ablation
        self._plot_ablation_temperature()

        # Plot memory quantity ablation
        self._plot_ablation_memory_quantity()

    def _plot_ablation_retrieve_k(self):
        """Plot retrieve_k ablation results."""
        csv_file = self.ablation_dir / "ablation_retrieve_k_swebench.csv"
        if not csv_file.exists():
            return

        data = self._load_csv(csv_file)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        k_values = [row["retrieve_k"] for row in data]
        resolve_rates = [row["resolve_rate"] * 100 for row in data]
        avg_steps = [row["avg_steps"] for row in data]

        # Resolve rate vs k
        ax1.plot(k_values, resolve_rates, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_xlabel('retrieve_k', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Effect of retrieve_k on Resolve Rate', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Average steps vs k
        ax2.plot(k_values, avg_steps, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax2.set_xlabel('retrieve_k', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Steps', fontsize=12, fontweight='bold')
        ax2.set_title('Effect of retrieve_k on Average Steps', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "ablation_retrieve_k_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def _plot_ablation_memory_source(self):
        """Plot memory source ablation results."""
        csv_file = self.ablation_dir / "ablation_memory_source_swebench.csv"
        if not csv_file.exists():
            return

        data = self._load_csv(csv_file)

        fig, ax = plt.subplots(figsize=(10, 6))

        sources = [row["memory_source"].replace("_", " ").title() for row in data]
        resolve_rates = [row["resolve_rate"] * 100 for row in data]

        bars = ax.bar(sources, resolve_rates, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Effect of Memory Source on Performance', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(resolve_rates) * 1.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "ablation_memory_source_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def _plot_ablation_temperature(self):
        """Plot temperature ablation results."""
        csv_file = self.ablation_dir / "ablation_temperature_swebench.csv"
        if not csv_file.exists():
            return

        data = self._load_csv(csv_file)

        fig, ax = plt.subplots(figsize=(10, 6))

        temps = [row["temperature"] for row in data]
        resolve_rates = [row["resolve_rate"] * 100 for row in data]

        ax.plot(temps, resolve_rates, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Extraction Temperature', fontsize=12, fontweight='bold')
        ax.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Effect of Extraction Temperature', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "ablation_temperature_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def _plot_ablation_memory_quantity(self):
        """Plot memory quantity ablation results."""
        csv_file = self.ablation_dir / "ablation_memory_quantity_swebench.csv"
        if not csv_file.exists():
            return

        data = self._load_csv(csv_file)

        fig, ax = plt.subplots(figsize=(10, 6))

        quantities = [row["max_items"] for row in data]
        resolve_rates = [row["resolve_rate"] * 100 for row in data]

        ax.plot(quantities, resolve_rates, marker='o', linewidth=2, markersize=8, color='#9b59b6')
        ax.set_xlabel('Max Items Per Trajectory', fontsize=12, fontweight='bold')
        ax.set_ylabel('Resolve Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Effect of Memory Quantity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "ablation_memory_quantity_swebench.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")

    def _load_csv(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Load CSV file and return as list of dictionaries."""
        data = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings to floats/ints
                converted_row = {}
                for key, value in row.items():
                    try:
                        if '.' in value:
                            converted_row[key] = float(value)
                        else:
                            converted_row[key] = int(value)
                    except (ValueError, AttributeError):
                        converted_row[key] = value
                data.append(converted_row)
        return data

    def generate_all_plots(self):
        """Generate all visualization plots."""
        logger.info("Generating all plots...")

        # Load results
        results = self.load_enhanced_results()

        if not results:
            logger.error("No results found to visualize")
            return

        # Generate main plots
        self.plot_overall_comparison(results)
        self.plot_difficulty_breakdown(results)
        self.plot_repository_analysis(results)
        self.plot_issue_type_analysis(results)
        self.plot_memory_analysis(results)

        # Generate ablation plots
        self.plot_ablation_studies()

        logger.info(f"All plots saved to {self.output_dir}")


def visualize_ablation_studies(ablation_dir: str, output_dir: str):
    """Visualize ablation studies (for external use)."""
    visualizer = SWEBenchVisualizer(
        results_dir="",
        ablation_dir=ablation_dir,
        output_dir=output_dir
    )
    visualizer.plot_ablation_studies()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for SWE-bench evaluation"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="comprehensive_results_swebench",
        help="Directory with enhanced results"
    )

    parser.add_argument(
        "--ablation-dir",
        type=str,
        default="ablation_results_swebench",
        help="Directory with ablation results"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations_swebench",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    visualizer = SWEBenchVisualizer(
        results_dir=args.results_dir,
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir
    )

    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
