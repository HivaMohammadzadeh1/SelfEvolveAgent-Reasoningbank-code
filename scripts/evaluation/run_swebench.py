#!/usr/bin/env python3
"""
Main evaluation script for SWE-bench (Table 2 reproduction).

Usage:
    # No-Memory baseline
    python run_swebench.py --mode no_memory

    # Synapse baseline
    python run_swebench.py --mode synapse

    # ReasoningBank
    python run_swebench.py --mode reasoningbank

    # Limit tasks for testing
    python run_swebench.py --mode reasoningbank --max_tasks 10
"""
import argparse
from pathlib import Path
import yaml
from loguru import logger
from dotenv import load_dotenv

from src.llm_client import create_llm_client
from src.embeddings import create_embedding_provider
from src.memory import ReasoningBank
from src.judge import TrajectoryJudge
from src.extractor import StrategyExtractor
from src.swebench_agent import SWEBenchAgent
from src.swebench_loader import load_swebench_dataset
from src.swebench_evaluator import SWEBenchEvaluator


def setup_logging(log_dir: str, mode: str):
    """Configure logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"swebench_{mode}.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )

    logger.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with ReasoningBank")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_memory", "reasoningbank"],
        required=True,
        help="Evaluation mode"
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
        help="Override max steps per task (default: 30)"
    )

    parser.add_argument(
        "--max_tasks",
        type=int,
        default=1,
        help="Maximum number of tasks to evaluate (default: 1 for quick testing, use --max_tasks 500 for full eval)"
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

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging - organize by mode
    log_dir = f"{config['paths']['logs_dir']}/swebench_{args.mode}"
    setup_logging(log_dir, args.mode)

    logger.info(f"Starting SWE-bench evaluation: mode={args.mode}, seed={args.seed}")
    logger.info(f"Config: {args.config}")
    logger.info(f"This will reproduce Table 2 from the ReasoningBank paper")

    # Create LLM client
    llm_provider = config["llm"]["provider"]
    llm_model = config["llm"]["model"]
    rate_limit_delay = config["llm"].get("rate_limit_delay", 1.0)
    logger.info(f"Using LLM: {llm_provider}/{llm_model} (rate limit delay: {rate_limit_delay}s)")
    llm_client = create_llm_client(llm_provider, llm_model, rate_limit_delay=rate_limit_delay)

    # Initialize components based on mode
    memory_bank = None
    judge = None
    extractor = None

    if args.mode == "reasoningbank":
        logger.info(f"Initializing {args.mode} components...")

        # Create embedding provider
        emb_provider = config["embedding"]["provider"]
        emb_model = config["embedding"]["model"]
        # Use embedding-specific rate limit if available, otherwise use LLM rate limit
        emb_rate_limit = config["embedding"].get("rate_limit_delay", rate_limit_delay)
        logger.info(f"Using embeddings: {emb_provider}/{emb_model} (rate limit delay: {emb_rate_limit}s)")

        embedding_provider = create_embedding_provider(emb_provider, emb_model, rate_limit_delay=emb_rate_limit)

        # Create memory bank
        bank_path = f"{config['paths']['bank_dir']}_swebench_{args.mode}"
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
        logger.info("ReasoningBank: Will extract strategies from successes AND failures")

        logger.info(f"{args.mode} components initialized")
    else:
        logger.info("Running in No-Memory baseline mode")

    # Create agent
    max_steps = args.max_steps or config["agent"]["max_steps"]
    timeout = config["agent"]["timeout_seconds"]

    agent = SWEBenchAgent(
        llm_client=llm_client,
        memory_bank=memory_bank,
        max_steps=max_steps,
        timeout=timeout
    )

    logger.info(f"Agent created: max_steps={max_steps}, timeout={timeout}s")

    # Create evaluator
    output_dir = args.output_dir or f"{config['paths']['results_dir']}/swebench_{args.mode}"
    checkpoint_interval = config["evaluation"]["checkpoint_interval"]

    evaluator = SWEBenchEvaluator(
        agent=agent,
        judge=judge,
        extractor=extractor,
        memory_bank=memory_bank,
        output_dir=output_dir,
        log_dir=log_dir,
        checkpoint_interval=checkpoint_interval,
        mode=args.mode
    )

    # Load SWE-Bench-Verified dataset
    logger.info("Loading SWE-Bench-Verified dataset...")
    dataset = load_swebench_dataset(
        cache_dir=args.cache_dir,
        split="test",
        max_tasks=args.max_tasks,
        start_index=args.start_index
    )

    logger.info(f"Dataset loaded: {len(dataset)} tasks")
    stats = dataset.get_stats()
    logger.info(f"Repository distribution:")
    for repo, count in list(stats["repos"].items())[:10]:  # Show top 10
        logger.info(f"  {repo}: {count}")

    # Run evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting SWE-bench evaluation: {args.mode}")
    logger.info(f"{'='*60}\n")

    logger.info("NOTE: This evaluation uses a simplified test harness.")
    logger.info("For official SWE-bench results, use the official evaluation harness:")
    logger.info("  https://www.swebench.com/SWE-bench/api/harness/")

    results = evaluator.evaluate_dataset(
        dataset=dataset,
        seed=args.seed,
        max_tasks=args.max_tasks,
        save_trajectories=True
    )

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"SWE-bench Results ({args.mode})")
    logger.info(f"{'='*60}")
    logger.info(f"Resolve Rate: {results.success_rate:.1%} ({results.successful_tasks}/{results.total_tasks})")
    logger.info(f"Avg Steps: {results.avg_steps:.1f}")
    logger.info(f"Total Time: {results.total_walltime:.1f}s")
    logger.info(f"Total Tokens: {results.total_tokens}")

    # Compare with paper results (Table 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Paper Results (Table 2) for reference:")
    logger.info(f"{'='*60}")

    if "gemini-2.5-flash" in llm_model:
        logger.info("Expected results (Gemini-2.5-flash):")
        logger.info("  No Memory:      34.2% resolve rate, 30.3 steps")
        logger.info("  Synapse:        35.4% resolve rate, 30.7 steps")
        logger.info("  ReasoningBank:  38.8% resolve rate, 27.5 steps")
    elif "gemini-2.5-pro" in llm_model:
        logger.info("Expected results (Gemini-2.5-pro):")
        logger.info("  No Memory:      54.0% resolve rate, 21.1 steps")
        logger.info("  Synapse:        53.4% resolve rate, 21.0 steps")
        logger.info("  ReasoningBank:  57.4% resolve rate, 19.8 steps")

    # Save final memory bank state
    if memory_bank:
        memory_bank.save_checkpoint()
        logger.info(f"\nFinal memory bank stats: {memory_bank.get_stats()}")

    logger.info("\nEvaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
