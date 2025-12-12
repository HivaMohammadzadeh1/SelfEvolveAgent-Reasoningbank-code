#!/usr/bin/env python3
"""
Main evaluation script for reproducing Table 1 results.

Usage:
    # No-Memory baseline
    python run_eval.py --mode no_memory --subset shopping
    
    # ReasoningBank
    python run_eval.py --mode reasoningbank --subset shopping
    
    # All subsets
    python run_eval.py --mode reasoningbank --subset all
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
from src.agent import ReActAgent
from src.evaluator import Evaluator, TaskDataset


def setup_logging(log_dir: str, mode: str, subset: str):
    """Configure logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{mode}_{subset}.log"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    logger.info(f"Logging to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run ReasoningBank evaluation")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["no_memory", "reasoningbank"],
        required=True,
        help="Evaluation mode"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="WebArena subset to evaluate (shopping, admin, gitlab, reddit, multi, or all)"
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
        help="Override max steps per task"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--use_real_browser",
        action="store_true",
        help="Use real WebArena browser environment (requires Docker setup)"
    )

    parser.add_argument(
        "--matts_mode",
        type=str,
        choices=["none", "parallel", "sequential"],
        default="none",
        help="MaTTS (Memory-aware Test-Time Scaling) mode from Paper Section 3.3 and Figure 4"
    )

    parser.add_argument(
        "--scaling_factor",
        type=int,
        default=1,
        help="Scaling factor k: number of trajectories (parallel) or refinement iterations (sequential). Paper tests k=1,2,3,4,5"
    )

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = config["paths"]["logs_dir"]
    setup_logging(log_dir, args.mode, args.subset)
    
    logger.info(f"Starting evaluation: mode={args.mode}, subset={args.subset}, seed={args.seed}")
    logger.info(f"Config: {args.config}")
    
    # Create LLM client
    llm_provider = config["llm"]["provider"]
    llm_model = config["llm"]["model"]
    logger.info(f"Using LLM: {llm_provider}/{llm_model}")
    llm_client = create_llm_client(llm_provider, llm_model)
    
    # Initialize components based on mode
    memory_bank = None
    judge = None
    extractor = None
    
    if args.mode == "reasoningbank":
        logger.info("Initializing ReasoningBank components...")
        
        # Create embedding provider
        emb_provider = config["embedding"]["provider"]
        emb_model = config["embedding"]["model"]
        logger.info(f"Using embeddings: {emb_provider}/{emb_model}")
        
        embedding_provider = create_embedding_provider(emb_provider, emb_model)
        
        # Create memory bank
        bank_path = config["paths"]["bank_dir"]
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
        
        logger.info("ReasoningBank components initialized")
    else:
        logger.info("Running in No-Memory baseline mode")
    
    # Create agent
    max_steps = args.max_steps or config["agent"]["max_steps"]
    timeout = config["agent"]["timeout_seconds"]
    
    agent = ReActAgent(
        llm_client=llm_client,
        memory_bank=memory_bank,
        max_steps=max_steps,
        timeout=timeout
    )
    
    logger.info(f"Agent created: max_steps={max_steps}, timeout={timeout}s")
    
    # Create evaluator
    output_dir = args.output_dir or config["paths"]["results_dir"]
    checkpoint_interval = config["evaluation"]["checkpoint_interval"]

    # MaTTS configuration (Paper Section 3.3, Figure 4)
    matts_mode = args.matts_mode
    scaling_factor = args.scaling_factor

    if matts_mode != "none":
        logger.info(f"MaTTS enabled: mode={matts_mode}, scaling_factor={scaling_factor}")
        if scaling_factor == 1:
            logger.warning("MaTTS mode enabled but scaling_factor=1. This is equivalent to no scaling.")

    evaluator = Evaluator(
        agent=agent,
        judge=judge,
        extractor=extractor,
        memory_bank=memory_bank,
        output_dir=output_dir,
        log_dir=log_dir,
        checkpoint_interval=checkpoint_interval,
        matts_mode=matts_mode,
        scaling_factor=scaling_factor
    )
    
    # Load dataset
    data_path = config["webarena"]["data_path"]
    subsets = config["webarena"]["subsets"]
    
    dataset = TaskDataset(data_path=data_path, subsets=subsets)
    
    # Determine which subsets to evaluate
    if args.subset == "all":
        eval_subsets = subsets
    else:
        if args.subset not in subsets:
            logger.error(f"Unknown subset: {args.subset}. Available: {subsets}")
            return
        eval_subsets = [args.subset]
    
    # Run evaluation
    logger.info(f"Evaluating subsets: {eval_subsets}")
    for subset in eval_subsets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating subset: {subset}")
        logger.info(f"{'='*60}\n")
        
        # Get use_real_browser from args or config
        use_real_browser = args.use_real_browser or config["agent"].get("use_real_browser", False)
        
        if use_real_browser:
            logger.warning("Real browser mode enabled - requires WebArena Docker environment")
        else:
            logger.warning("Using mock browser - page interactions are simulated")
            logger.info("Real WebArena tasks are loaded, but browser is mocked")
            logger.info("To use real browser: set use_real_browser: true in config.yaml")
        
        results = evaluator.evaluate_dataset(
            dataset=dataset,
            subset=subset,
            seed=args.seed,
            save_trajectories=True,
            use_real_browser=use_real_browser
        )
        
        logger.info(f"\nResults for {subset}:")
        logger.info(f"  Success Rate: {results.success_rate:.3f}")
        logger.info(f"  Avg Steps: {results.avg_steps:.1f}")
        logger.info(f"  Successful: {results.successful_tasks}/{results.total_tasks}")
        logger.info(f"  Total tokens: {results.total_tokens}")
        logger.info(f"  Total time: {results.total_walltime:.1f}s")
    
    # Save final memory bank state
    if memory_bank:
        memory_bank.save_checkpoint()
        logger.info(f"Final memory bank stats: {memory_bank.get_stats()}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
