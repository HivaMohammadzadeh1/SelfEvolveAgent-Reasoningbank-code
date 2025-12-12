#!/usr/bin/env python3
"""
Main evaluation script for Mind2Web (Table 3 reproduction).

Usage:
    # No-Memory baseline - Cross-Task split
    python run_mind2web.py --mode no_memory --split test_task

    # ReasoningBank - Cross-Website split
    python run_mind2web.py --mode reasoningbank --split test_website

    # ReasoningBank - Cross-Domain split
    python run_mind2web.py --mode reasoningbank --split test_domain

    # Limit tasks for testing
    python run_mind2web.py --mode reasoningbank --split test_task --max_tasks 10
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
from loguru import logger
from dotenv import load_dotenv

# Fix for OpenMP library conflict on macOS (required for FAISS)
# This prevents "OMP: Error #15: Initializing libomp.dylib" crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.llm_client import create_llm_client
from src.embeddings import create_embedding_provider
from src.memory import ReasoningBank
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.mind2web_agent import Mind2WebAgent
from src.mind2web_loader import load_mind2web_dataset
from src.mind2web_evaluator import Mind2WebEvaluator


def setup_logging(log_dir: str, mode: str, split: str):
    """Configure logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"mind2web_{mode}_{split}.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )

    logger.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Mind2Web evaluation with ReasoningBank")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_memory", "synapse", "awm", "reasoningbank"],
        required=True,
        help="Evaluation mode"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["test_task", "test_website", "test_domain"],
        default="test_task",
        help="Mind2Web test split: test_task (Cross-Task, 252), test_website (Cross-Website, 177), test_domain (Cross-Domain, 912)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max steps per task (default: 15)"
    )

    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (default: all tasks in split)"
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for task evaluation (for parallel processing)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for dataset"
    )

    parser.add_argument(
        "--use_real_browser",
        action="store_true",
        help="Use real browser environment (requires BrowserGym setup)"
    )

    parser.add_argument(
        "--exclude_blocked",
        action="store_true",
        help="Exclude websites with anti-bot protection"
    )

    parser.add_argument(
        "--only_working",
        action="store_true",
        help="Only use confirmed working websites (budget, eventbrite, tvguide, underarmour, soundcloud)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging
    log_dir = config["paths"]["logs_dir"]
    setup_logging(log_dir, args.mode, args.split)

    # Split names for display
    split_names = {
        "test_task": "Cross-Task (252 tasks)",
        "test_website": "Cross-Website (177 tasks)",
        "test_domain": "Cross-Domain (912 tasks)"
    }

    logger.info(f"Starting Mind2Web evaluation: mode={args.mode}, split={split_names[args.split]}, seed={args.seed}")
    logger.info(f"Config: {args.config}")
    logger.info(f"This will reproduce Table 3 from the ReasoningBank paper")

    # Create LLM client
    llm_provider = config["llm"]["provider"]
    llm_model = config["llm"]["model"]
    logger.info(f"Using LLM: {llm_provider}/{llm_model}")
    llm_client = create_llm_client(llm_provider, llm_model)

    # Initialize components based on mode
    memory_bank = None
    judge = None
    extractor = None

    if args.mode in ["synapse", "awm", "reasoningbank"]:
        logger.info(f"Initializing {args.mode} components...")

        # Create embedding provider
        emb_provider = config["embedding"]["provider"]
        emb_model = config["embedding"]["model"]
        logger.info(f"Using embeddings: {emb_provider}/{emb_model}")

        embedding_provider = create_embedding_provider(emb_provider, emb_model)

        # Create memory bank
        bank_path = f"{config['paths']['bank_dir']}_mind2web_{args.mode}_{args.split}"
        dedup_threshold = config["memory"]["dedup_threshold"]

        memory_bank = ReasoningBank(
            bank_path=bank_path,
            embedding_provider=embedding_provider,
            dedup_threshold=dedup_threshold
        )

        logger.info(f"Memory bank initialized: {memory_bank.get_stats()}")

        # Create judge
        judge = TrajectoryJudge(
            llm_client=llm_client,
            temperature=config["llm"]["judge_temperature"]
        )

        # Create extractor
        extractor = StrategyExtractor(
            llm_client=llm_client,
            temperature=config["llm"]["extractor_temperature"],
            max_items=config["memory"]["max_items_per_trajectory"]
        )

        if args.mode == "reasoningbank":
            logger.info("ReasoningBank: Will extract strategies from successes AND failures")
        elif args.mode == "synapse":
            logger.info("Synapse: Will extract strategies from successes only")
        elif args.mode == "awm":
            logger.info("AWM: Will use action-weighted memory retrieval")

        logger.info(f"{args.mode} components initialized")
    else:
        logger.info("Running in No-Memory baseline mode")

    # Create agent
    max_steps = args.max_steps or 15  # Mind2Web typically uses fewer steps than WebArena
    timeout = config["agent"]["timeout_seconds"]

    # Use config value if command line arg not explicitly provided
    use_real_browser = args.use_real_browser if '--use_real_browser' in sys.argv else config["agent"].get("use_real_browser", False)

    agent = Mind2WebAgent(
        llm_client=llm_client,
        memory_bank=memory_bank,
        max_steps=max_steps,
        timeout=timeout,
        use_real_browser=use_real_browser
    )

    logger.info(f"Agent created: max_steps={max_steps}, timeout={timeout}s")
    if use_real_browser:
        logger.info("Using real browser environment (BrowserGym)")
    else:
        logger.info("Using mock browser environment (for testing)")

    # Create evaluator
    output_dir = args.output_dir or f"{config['paths']['results_dir']}/mind2web_{args.mode}_{args.split}"
    checkpoint_interval = config["evaluation"]["checkpoint_interval"]

    evaluator = Mind2WebEvaluator(
        agent=agent,
        judge=judge,
        extractor=extractor,
        memory_bank=memory_bank,
        output_dir=output_dir,
        log_dir=log_dir,
        checkpoint_interval=checkpoint_interval,
        mode=args.mode
    )

    # Load Mind2Web dataset
    logger.info(f"Loading Mind2Web dataset: {split_names[args.split]}...")
    dataset = load_mind2web_dataset(
        split=args.split,
        cache_dir=args.cache_dir,
        max_tasks=args.max_tasks,
        start_index=args.start_index,
        exclude_blocked=args.exclude_blocked,
        only_working=args.only_working
    )

    logger.info(f"Dataset loaded: {len(dataset)} tasks")
    stats = dataset.get_stats()
    logger.info(f"Split: {stats['split']}")
    logger.info(f"Expected count: {stats['expected_count']}")
    logger.info(f"Websites: {stats['num_websites']}, Domains: {stats['num_domains']}")

    # Run evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Mind2Web evaluation: {args.mode} on {split_names[args.split]}")
    logger.info(f"{'='*60}\n")

    results = evaluator.evaluate_dataset(
        dataset=dataset,
        seed=args.seed,
        max_tasks=args.max_tasks,
        save_trajectories=True
    )

    # Log results with Mind2Web metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"Mind2Web Results ({args.mode} - {split_names[args.split]})")
    logger.info(f"{'='*60}")

    # Extract Mind2Web metrics
    mind2web_metrics = results.mind2web_metrics if hasattr(results, 'mind2web_metrics') else {}
    ea = mind2web_metrics.get("element_accuracy", 0.0) * 100
    af1 = mind2web_metrics.get("action_f1", 0.0) * 100
    ssr = mind2web_metrics.get("step_success_rate", 0.0) * 100
    sr = mind2web_metrics.get("task_success_rate", 0.0) * 100

    logger.info(f"Element Accuracy (EA):    {ea:.1f}%")
    logger.info(f"Action F1 (AF1):          {af1:.1f}%")
    logger.info(f"Step Success Rate (SSR):  {ssr:.1f}%")
    logger.info(f"Success Rate (SR):        {sr:.1f}%")
    logger.info(f"Avg Steps: {results.avg_steps:.1f}")
    logger.info(f"Total Time: {results.total_walltime:.1f}s")
    logger.info(f"Total Tokens: {results.total_tokens}")

    # Compare with paper results (Table 3)
    logger.info(f"\n{'='*60}")
    logger.info(f"Paper Results (Table 3) for reference:")
    logger.info(f"{'='*60}")

    # Expected results from Table 3 (format: EA/AF1/SSR/SR)
    if "gemini-2.5-flash" in llm_model.lower() or "gemini-flash" in llm_model.lower():
        logger.info("Expected results (Gemini-2.5-flash):")
        logger.info("  Cross-Task (252):")
        logger.info("    No Memory:      47.9 / 55.0 / 40.2 / 3.2")
        logger.info("    Synapse:        49.2 / 56.0 / 41.8 / 3.6")
        logger.info("    AWM:            50.1 / 57.8 / 42.9 / 3.6")
        logger.info("    ReasoningBank:  52.1 / 60.4 / 44.9 / 4.8")
        logger.info("  Cross-Website (177):")
        logger.info("    No Memory:      41.8 / 50.3 / 35.0 / 2.8")
        logger.info("    Synapse:        42.5 / 51.1 / 36.2 / 2.8")
        logger.info("    AWM:            43.4 / 52.6 / 37.3 / 3.4")
        logger.info("    ReasoningBank:  45.2 / 54.9 / 39.0 / 4.0")
        logger.info("  Cross-Domain (912):")
        logger.info("    No Memory:      36.5 / 45.8 / 30.1 / 2.2")
        logger.info("    Synapse:        37.8 / 46.9 / 31.6 / 2.4")
        logger.info("    AWM:            38.1 / 48.2 / 32.4 / 2.4")
        logger.info("    ReasoningBank:  40.3 / 50.7 / 34.5 / 2.9")
    elif "gemini-2.5-pro" in llm_model.lower() or "gemini-pro" in llm_model.lower():
        logger.info("Expected results (Gemini-2.5-pro):")
        logger.info("  Cross-Task (252):")
        logger.info("    No Memory:      58.2 / 66.1 / 51.3 / 6.0")
        logger.info("    Synapse:        59.4 / 67.5 / 52.8 / 6.7")
        logger.info("    AWM:            60.1 / 68.9 / 53.7 / 7.1")
        logger.info("    ReasoningBank:  63.8 / 72.3 / 57.4 / 8.3")
        logger.info("  Cross-Website (177):")
        logger.info("    No Memory:      53.7 / 62.8 / 47.5 / 5.1")
        logger.info("    Synapse:        54.2 / 63.6 / 48.0 / 5.1")
        logger.info("    AWM:            55.6 / 64.9 / 49.7 / 5.6")
        logger.info("    ReasoningBank:  58.8 / 68.5 / 52.5 / 6.8")
        logger.info("  Cross-Domain (912):")
        logger.info("    No Memory:      48.9 / 58.2 / 42.3 / 3.8")
        logger.info("    Synapse:        49.5 / 59.1 / 43.1 / 3.9")
        logger.info("    AWM:            50.7 / 60.8 / 44.6 / 4.2")
        logger.info("    ReasoningBank:  53.4 / 64.2 / 47.8 / 5.3")

    logger.info(f"\nNote: Format is EA / AF1 / SSR / SR")
    logger.info(f"Your results: {ea:.1f} / {af1:.1f} / {ssr:.1f} / {sr:.1f}")

    # Save final memory bank state
    if memory_bank:
        memory_bank.save_checkpoint()
        logger.info(f"\nFinal memory bank stats: {memory_bank.get_stats()}")

    logger.info("\nEvaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
