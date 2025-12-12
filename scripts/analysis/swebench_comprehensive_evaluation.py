#!/usr/bin/env python3
"""
Comprehensive evaluation script for SWE-bench.

Enhances basic results with:
- Task difficulty classification
- Repository analysis
- Issue type classification
- Retrieval quality metrics
- Memory content analysis
- Comprehensive reporting
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
import yaml

from src.swebench_task_analyzer import (
    SWEBenchTaskAnalyzer,
    classify_task_difficulty,
    classify_issue_type
)


class SWEBenchComprehensiveEvaluator:
    """Comprehensive evaluator for SWE-bench results."""

    def __init__(
        self,
        results_dir: str,
        output_dir: str,
        config: Dict[str, Any]
    ):
        """
        Initialize evaluator.

        Args:
            results_dir: Directory containing basic results
            output_dir: Directory for enhanced results
            config: Configuration dictionary
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.config = config

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize task analyzer
        self.analyzer = SWEBenchTaskAnalyzer()

    def enhance_results(self, mode: str) -> Dict[str, Any]:
        """
        Enhance results for a given mode with comprehensive metrics.

        Args:
            mode: Evaluation mode (no_memory, reasoningbank)

        Returns:
            Enhanced results dictionary
        """
        logger.info(f"Enhancing results for mode: {mode}")

        # Load basic results
        results_file = self.results_dir / mode / "results.json"
        summary_file = self.results_dir / mode / "summary.json"

        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}

        with open(results_file, "r") as f:
            results = json.load(f)

        with open(summary_file, "r") as f:
            summary = json.load(f)

        logger.info(f"Loaded {len(results)} task results")

        # Enhance with task analysis
        enhanced_results = {
            "mode": mode,
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r.get("success", False)),
            "resolve_rate": summary.get("success_rate", 0.0),
            "avg_steps": summary.get("avg_steps", 0.0),
            "total_tokens": summary.get("total_tokens", 0),
            "total_walltime": summary.get("total_walltime", 0.0),
            "task_results": results
        }

        # Add difficulty breakdown
        enhanced_results["difficulty_breakdown"] = self._analyze_difficulty(results)

        # Add repository breakdown
        enhanced_results["repository_breakdown"] = self._analyze_repositories(results)

        # Add issue type breakdown
        enhanced_results["issue_type_breakdown"] = self._analyze_issue_types(results)

        # Add retrieval quality metrics (if applicable)
        if mode in ["synapse", "reasoningbank"]:
            enhanced_results["retrieval_quality"] = self._analyze_retrieval_quality(results)

        # Add memory metrics (if applicable)
        if mode in ["synapse", "reasoningbank"]:
            enhanced_results["memory_metrics"] = self._analyze_memory_metrics(mode)

        # Add step efficiency analysis
        enhanced_results["efficiency_metrics"] = self._analyze_efficiency(results)

        return enhanced_results

    def _analyze_difficulty(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by task difficulty."""
        logger.info("Analyzing task difficulty...")

        difficulty_stats = {
            "easy": {"count": 0, "successful": 0, "total_steps": 0},
            "medium": {"count": 0, "successful": 0, "total_steps": 0},
            "hard": {"count": 0, "successful": 0, "total_steps": 0}
        }

        for result in results:
            # Classify difficulty based on task complexity
            task_id = result.get("task_id", "")
            difficulty = classify_task_difficulty(result)

            difficulty_stats[difficulty]["count"] += 1

            if result.get("success", False):
                difficulty_stats[difficulty]["successful"] += 1
                difficulty_stats[difficulty]["total_steps"] += result.get("steps", 0)

        # Compute metrics for each difficulty
        breakdown = {}
        for difficulty, stats in difficulty_stats.items():
            count = stats["count"]
            successful = stats["successful"]
            resolve_rate = successful / count if count > 0 else 0.0
            avg_steps = stats["total_steps"] / successful if successful > 0 else 0.0

            breakdown[difficulty] = {
                "count": count,
                "successful": successful,
                "resolve_rate": resolve_rate,
                "avg_steps": avg_steps
            }

        logger.info(f"Difficulty breakdown: {breakdown}")
        return breakdown

    def _analyze_repositories(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by repository."""
        logger.info("Analyzing repositories...")

        repo_stats = {}

        for result in results:
            # Extract repository from task_id (format: repo_name-issue_number)
            task_id = result.get("task_id", "")
            repo = self._extract_repo_from_task_id(task_id)

            if repo not in repo_stats:
                repo_stats[repo] = {
                    "count": 0,
                    "successful": 0,
                    "total_steps": 0
                }

            repo_stats[repo]["count"] += 1

            if result.get("success", False):
                repo_stats[repo]["successful"] += 1
                repo_stats[repo]["total_steps"] += result.get("steps", 0)

        # Compute metrics for each repository
        breakdown = {}
        for repo, stats in repo_stats.items():
            count = stats["count"]
            successful = stats["successful"]
            resolve_rate = successful / count if count > 0 else 0.0
            avg_steps = stats["total_steps"] / successful if successful > 0 else 0.0

            breakdown[repo] = {
                "count": count,
                "successful": successful,
                "resolve_rate": resolve_rate,
                "avg_steps": avg_steps
            }

        # Sort by count (most common repositories first)
        breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1]["count"], reverse=True))

        logger.info(f"Repository breakdown: {len(breakdown)} repositories")
        return breakdown

    def _analyze_issue_types(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by issue type."""
        logger.info("Analyzing issue types...")

        type_stats = {
            "bug_fix": {"count": 0, "successful": 0, "total_steps": 0},
            "feature_request": {"count": 0, "successful": 0, "total_steps": 0},
            "performance": {"count": 0, "successful": 0, "total_steps": 0},
            "documentation": {"count": 0, "successful": 0, "total_steps": 0},
            "other": {"count": 0, "successful": 0, "total_steps": 0}
        }

        for result in results:
            # Classify issue type
            issue_type = classify_issue_type(result)

            type_stats[issue_type]["count"] += 1

            if result.get("success", False):
                type_stats[issue_type]["successful"] += 1
                type_stats[issue_type]["total_steps"] += result.get("steps", 0)

        # Compute metrics for each type
        breakdown = {}
        for issue_type, stats in type_stats.items():
            count = stats["count"]
            successful = stats["successful"]
            resolve_rate = successful / count if count > 0 else 0.0
            avg_steps = stats["total_steps"] / successful if successful > 0 else 0.0

            breakdown[issue_type] = {
                "count": count,
                "successful": successful,
                "resolve_rate": resolve_rate,
                "avg_steps": avg_steps
            }

        logger.info(f"Issue type breakdown: {breakdown}")
        return breakdown

    def _analyze_retrieval_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retrieval quality metrics."""
        logger.info("Analyzing retrieval quality...")

        total_retrievals = 0
        successful_retrievals = 0
        similarity_scores = []

        for result in results:
            # Check if retrieval was performed
            # Note: This assumes trajectory data includes retrieval info
            trajectory_path = result.get("trajectory_path")
            if trajectory_path and Path(trajectory_path).exists():
                with open(trajectory_path, "r") as f:
                    trajectory = json.load(f)

                # Check for retrieved memories
                retrieved = trajectory.get("retrieved_memories", [])
                if retrieved:
                    total_retrievals += 1

                    # Check if retrieval helped (task succeeded)
                    if result.get("success", False):
                        successful_retrievals += 1

                    # Collect similarity scores
                    for memory in retrieved:
                        if "similarity" in memory:
                            similarity_scores.append(memory["similarity"])

        # Compute metrics
        precision = successful_retrievals / total_retrievals if total_retrievals > 0 else 0.0
        coverage = total_retrievals / len(results) if len(results) > 0 else 0.0

        retrieval_quality = {
            "precision": {
                "overall": precision,
                "successful_retrievals": successful_retrievals,
                "total_retrievals": total_retrievals
            },
            "coverage": {
                "coverage": coverage,
                "tasks_with_retrieval": total_retrievals,
                "total_tasks": len(results)
            }
        }

        if similarity_scores:
            import statistics
            retrieval_quality["similarity_stats"] = {
                "mean": statistics.mean(similarity_scores),
                "std": statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                "min": min(similarity_scores),
                "max": max(similarity_scores)
            }

        logger.info(f"Retrieval quality: precision={precision:.2%}, coverage={coverage:.2%}")
        return retrieval_quality

    def _analyze_memory_metrics(self, mode: str) -> Dict[str, Any]:
        """Analyze memory bank metrics."""
        logger.info("Analyzing memory metrics...")

        # Load memory bank
        bank_path = Path(f"memory_bank/swebench_{mode}")

        if not bank_path.exists():
            logger.warning(f"Memory bank not found: {bank_path}")
            return {}

        # Load memory bank metadata
        metadata_file = bank_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            return {
                "quantity": {
                    "total_memories": metadata.get("total_memories", 0),
                    "success_memories": metadata.get("success_memories", 0),
                    "failure_memories": metadata.get("failure_memories", 0),
                    "success_ratio": metadata.get("success_ratio", 0.0)
                },
                "quality": {
                    "avg_content_length": metadata.get("avg_content_length", 0),
                    "unique_strategies": metadata.get("unique_strategies", 0),
                    "uniqueness_ratio": metadata.get("uniqueness_ratio", 0.0)
                }
            }

        return {}

    def _analyze_efficiency(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze efficiency metrics."""
        logger.info("Analyzing efficiency...")

        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {}

        steps = [r.get("steps", 0) for r in successful_results]
        walltimes = [r.get("walltime", 0.0) for r in successful_results]

        import statistics

        return {
            "steps": {
                "mean": statistics.mean(steps),
                "std": statistics.stdev(steps) if len(steps) > 1 else 0.0,
                "min": min(steps),
                "max": max(steps),
                "median": statistics.median(steps)
            },
            "walltime": {
                "mean": statistics.mean(walltimes),
                "std": statistics.stdev(walltimes) if len(walltimes) > 1 else 0.0,
                "min": min(walltimes),
                "max": max(walltimes),
                "median": statistics.median(walltimes)
            }
        }

    def _extract_repo_from_task_id(self, task_id: str) -> str:
        """Extract repository name from task ID."""
        # Task IDs are typically in format: owner__repo-issue_number
        # or owner-repo-issue_number
        parts = task_id.replace("__", "-").split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return "unknown"

    def generate_comparison_report(
        self,
        enhanced_results: Dict[str, Dict[str, Any]]
    ):
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")

        report_path = self.output_dir / "comprehensive_report_swebench.md"

        with open(report_path, "w") as f:
            f.write("# SWE-bench Comprehensive Evaluation Report\n\n")

            # Overall comparison
            f.write("## Overall Performance\n\n")
            f.write("| Method | Resolve Rate | Avg Steps | Total Tasks |\n")
            f.write("|--------|--------------|-----------|-------------|\n")

            for mode in ["no_memory", "reasoningbank"]:
                if mode in enhanced_results:
                    results = enhanced_results[mode]
                    resolve_rate = results.get("resolve_rate", 0.0)
                    avg_steps = results.get("avg_steps", 0.0)
                    total = results.get("total_tasks", 0)

                    f.write(f"| {mode} | {resolve_rate:.1%} | {avg_steps:.1f} | {total} |\n")

            f.write("\n")

            # Difficulty breakdown
            f.write("## Performance by Difficulty\n\n")

            for mode in ["no_memory", "reasoningbank"]:
                if mode in enhanced_results:
                    f.write(f"### {mode}\n\n")
                    f.write("| Difficulty | Count | Resolve Rate | Avg Steps |\n")
                    f.write("|------------|-------|--------------|------------|\n")

                    difficulty_breakdown = enhanced_results[mode].get("difficulty_breakdown", {})
                    for difficulty in ["easy", "medium", "hard"]:
                        if difficulty in difficulty_breakdown:
                            stats = difficulty_breakdown[difficulty]
                            f.write(f"| {difficulty} | {stats['count']} | {stats['resolve_rate']:.1%} | {stats['avg_steps']:.1f} |\n")

                    f.write("\n")

            # Repository breakdown (top 10)
            f.write("## Performance by Repository (Top 10)\n\n")

            if "reasoningbank" in enhanced_results:
                repo_breakdown = enhanced_results["reasoningbank"].get("repository_breakdown", {})
                f.write("| Repository | Count | Resolve Rate | Avg Steps |\n")
                f.write("|------------|-------|--------------|------------|\n")

                for repo, stats in list(repo_breakdown.items())[:10]:
                    f.write(f"| {repo} | {stats['count']} | {stats['resolve_rate']:.1%} | {stats['avg_steps']:.1f} |\n")

                f.write("\n")

            # Issue type breakdown
            f.write("## Performance by Issue Type\n\n")

            if "reasoningbank" in enhanced_results:
                type_breakdown = enhanced_results["reasoningbank"].get("issue_type_breakdown", {})
                f.write("| Issue Type | Count | Resolve Rate | Avg Steps |\n")
                f.write("|------------|-------|--------------|------------|\n")

                for issue_type, stats in type_breakdown.items():
                    if stats['count'] > 0:
                        f.write(f"| {issue_type} | {stats['count']} | {stats['resolve_rate']:.1%} | {stats['avg_steps']:.1f} |\n")

                f.write("\n")

            # Memory metrics
            f.write("## Memory Metrics\n\n")

            for mode in ["reasoningbank"]:
                if mode in enhanced_results:
                    memory_metrics = enhanced_results[mode].get("memory_metrics", {})
                    if memory_metrics:
                        f.write(f"### {mode}\n\n")

                        quantity = memory_metrics.get("quantity", {})
                        f.write(f"- Total memories: {quantity.get('total_memories', 0)}\n")
                        f.write(f"- Success/Failure ratio: {quantity.get('success_ratio', 0.0):.1%}\n")

                        quality = memory_metrics.get("quality", {})
                        f.write(f"- Unique strategies: {quality.get('unique_strategies', 0)}\n")
                        f.write(f"- Uniqueness ratio: {quality.get('uniqueness_ratio', 0.0):.1%}\n")
                        f.write("\n")

        logger.info(f"Report saved to {report_path}")

    def run(self):
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")

        # Enhance results for each mode
        enhanced_results = {}

        for mode in ["no_memory", "synapse", "reasoningbank"]:
            if (self.results_dir / mode).exists():
                enhanced_results[mode] = self.enhance_results(mode)

                # Save enhanced results
                output_file = self.output_dir / f"{mode}_swebench_enhanced.json"
                with open(output_file, "w") as f:
                    json.dump(enhanced_results[mode], f, indent=2)

                logger.info(f"Enhanced results saved to {output_file}")

        # Generate comparison report
        if enhanced_results:
            self.generate_comparison_report(enhanced_results)

            # Save comparison data
            comparison_file = self.output_dir / "comparisons.json"
            with open(comparison_file, "w") as f:
                json.dump(enhanced_results, f, indent=2)

        logger.info("Comprehensive evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for SWE-bench results"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.swebench.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="comprehensive_results_swebench",
        help="Directory containing basic results"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="comprehensive_results_swebench",
        help="Output directory for enhanced results"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create evaluator
    evaluator = SWEBenchComprehensiveEvaluator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        config=config
    )

    # Run evaluation
    evaluator.run()


if __name__ == "__main__":
    main()
