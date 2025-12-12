#!/usr/bin/env python3
"""
Reproduce Table 1 from ReasoningBank paper.

This script runs the exact experiments needed to reproduce Table 1:
- No Memory (baseline)
- ReasoningBank
- Across all 5 subsets: Shopping, Admin, Gitlab, Reddit, Multi
- With specified models: Gemini-2.5-flash, Gemini-2.5-pro, Claude-3.7-sonnet

Usage:
    # Run with Gemini 2.5 Flash (recommended)
    python reproduce_table1.py --model gemini-2.5-flash
    
    # Run with Gemini 2.5 Pro  
    python reproduce_table1.py --model gemini-2.5-pro
    
    # Run specific subset only
    python reproduce_table1.py --model gemini-2.5-flash --subset admin
    
    # Run both modes (No Memory + ReasoningBank)
    python reproduce_table1.py --model gemini-2.5-flash --full
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from loguru import logger
import pandas as pd
import json


# Expected task counts from paper
EXPECTED_COUNTS = {
    "shopping": 187,
    "admin": 182,
    "gitlab": 180,
    "reddit": 106,
    "multi": 29,
}

# WebArena environment variables
WEBARENA_ENV_VARS = [
    "SHOPPING",
    "SHOPPING_ADMIN", 
    "REDDIT",
    "GITLAB",
    "MAP",
    "WIKIPEDIA",
    "HOMEPAGE"
]

# Model configurations
MODEL_CONFIGS = {
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash"
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "model": "gemini-2.5-pro"
    },
    "claude-3.7-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022"
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o"
    },
    # TogetherAI models (open-source alternatives)
    "deepseek-r1": {
        "provider": "together",
        "model": "deepseek-ai/DeepSeek-R1"
    },
    "llama-4-maverick": {
        "provider": "together",
        "model": "meta-llama/Llama-4-Maverick"
    },
    "qwen3-235b": {
        "provider": "together",
        "model": "Qwen/Qwen3-235B-Instruct"
    },
    "gemma-3-27b": {
        "provider": "together",
        "model": "google/gemma-3-27b-it"
    }
}


def check_webarena_urls():
    """Validate that WebArena environment URLs are set and accessible."""
    logger.info("Checking WebArena environment variables...")
    
    missing = []
    urls_to_test = {}
    
    for var in WEBARENA_ENV_VARS:
        value = os.environ.get(var)
        if not value:
            missing.append(var)
        else:
            logger.info(f"  ✓ {var}: {value}")
            urls_to_test[var] = value
    
    if missing:
        logger.error(f"Missing WebArena environment variables: {missing}")
        logger.error("Please set them in your shell before running this script")
        logger.error("Example: export SHOPPING='http://your-host:7770'")
        return False
    
    logger.info("✓ All WebArena URLs are set")
    
    # Quick connectivity test to SHOPPING (main site)
    try:
        import requests
        test_url = urls_to_test.get("SHOPPING")
        if test_url:
            logger.info(f"Testing connectivity to {test_url}...")
            response = requests.get(test_url, timeout=10)
            logger.info(f"✓ WebArena is accessible (status: {response.status_code})")
    except ImportError:
        logger.warning("requests library not available, skipping connectivity test")
    except Exception as e:
        logger.warning(f"Could not connect to WebArena: {e}")
        logger.warning("The URLs are set but may not be accessible")
        logger.warning("Make sure WebArena containers are running!")
    
    return True


def run_evaluation(mode: str, subset: str, model_name: str, seed: int = 42):
    """
    Run evaluation for a specific mode and subset.
    
    Args:
        mode: "no_memory" or "reasoningbank"
        subset: Subset name (shopping, admin, etc.)
        model_name: Model identifier
        seed: Random seed
    """
    logger.info(f"Running {mode} on {subset} with {model_name} (seed={seed})")
    
    # Get model config
    config = MODEL_CONFIGS[model_name]
    
    # Build command
    cmd = [
        "python", "run_eval.py",
        "--mode", mode,
        "--subset", subset,
        "--seed", str(seed),
        "--config", "config.yaml"
    ]
    
    # Run evaluation
    try:
        print(cmd)
        print("running run_eval.py")
        result = subprocess.run(
            cmd,
            check=True,
            env=os.environ.copy()  # Explicitly pass environment variables
        )
        logger.info(f"✓ Completed {mode}/{subset}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed {mode}/{subset}")
        logger.error(f"Error: {e}")
        return False


def load_results(mode: str, subset: str) -> dict:
    """Load results from JSON file."""
    result_file = Path(f"results/{mode}_{subset}.json")
    
    if not result_file.exists():
        logger.warning(f"Results file not found: {result_file}")
        return None
    
    with open(result_file, "r") as f:
        return json.load(f)


def generate_table1(model_name: str):
    """
    Generate Table 1 comparison from results.
    
    Args:
        model_name: Model used for evaluation
    """
    logger.info(f"\nGenerating Table 1 for {model_name}")
    
    subsets = ["shopping", "admin", "gitlab", "reddit", "multi"]
    
    rows = []
    
    for subset in subsets:
        # Load results
        no_mem = load_results("no_memory", subset)
        rb = load_results("reasoningbank", subset)
        
        if not no_mem or not rb:
            logger.warning(f"Missing results for {subset}, skipping")
            continue
        
        row = {
            "Subset": subset.capitalize(),
            "Count": EXPECTED_COUNTS[subset],
            "No-Mem SR": no_mem["success_rate"] * 100,
            "No-Mem Steps": no_mem["avg_steps"],
            "RB SR": rb["success_rate"] * 100,
            "RB Steps": rb["avg_steps"],
            "ΔSR": (rb["success_rate"] - no_mem["success_rate"]) * 100,
            "ΔSteps": rb["avg_steps"] - no_mem["avg_steps"]
        }
        rows.append(row)
    
    # Calculate overall
    if rows:
        total_tasks = sum(EXPECTED_COUNTS.values())
        
        # Weighted averages
        overall_no_mem_sr = sum(r["No-Mem SR"] * EXPECTED_COUNTS[r["Subset"].lower()] for r in rows) / total_tasks
        overall_rb_sr = sum(r["RB SR"] * EXPECTED_COUNTS[r["Subset"].lower()] for r in rows) / total_tasks
        overall_no_mem_steps = sum(r["No-Mem Steps"] * EXPECTED_COUNTS[r["Subset"].lower()] for r in rows) / total_tasks
        overall_rb_steps = sum(r["RB Steps"] * EXPECTED_COUNTS[r["Subset"].lower()] for r in rows) / total_tasks
        
        overall_row = {
            "Subset": "Overall",
            "Count": total_tasks,
            "No-Mem SR": overall_no_mem_sr,
            "No-Mem Steps": overall_no_mem_steps,
            "RB SR": overall_rb_sr,
            "RB Steps": overall_rb_steps,
            "ΔSR": overall_rb_sr - overall_no_mem_sr,
            "ΔSteps": overall_rb_steps - overall_no_mem_steps
        }
        rows.append(overall_row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Format for display
    df["No-Mem SR"] = df["No-Mem SR"].apply(lambda x: f"{x:.1f}")
    df["RB SR"] = df["RB SR"].apply(lambda x: f"{x:.1f}")
    df["No-Mem Steps"] = df["No-Mem Steps"].apply(lambda x: f"{x:.1f}")
    df["RB Steps"] = df["RB Steps"].apply(lambda x: f"{x:.1f}")
    df["ΔSR"] = df["ΔSR"].apply(lambda x: f"{x:+.1f}")
    df["ΔSteps"] = df["ΔSteps"].apply(lambda x: f"{x:+.1f}")
    
    # Print table
    print("\n" + "="*80)
    print(f"TABLE 1 REPRODUCTION - {model_name}")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    output_file = f"results/table1_{model_name.replace('.', '_')}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"\nSaved to: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 from ReasoningBank paper"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "shopping", "admin", "gitlab", "reddit", "multi"],
        help="Subset to evaluate (default: all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "no_memory", "reasoningbank"],
        help="Which mode to run (default: both)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full experiment (both modes, all subsets)"
    )
    
    args = parser.parse_args()
    
    # Check WebArena URLs are set
    if not check_webarena_urls():
        logger.error("WebArena environment not configured properly!")
        sys.exit(1)
    
    # Update config with selected model
    model_config = MODEL_CONFIGS[args.model]
    logger.info(f"Using model: {args.model}")
    logger.info(f"  Provider: {model_config['provider']}")
    logger.info(f"  Model: {model_config['model']}")
    
    # Update config.yaml with selected model
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["llm"]["provider"] = model_config["provider"]
    config["llm"]["model"] = model_config["model"]
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Updated config.yaml with selected model")
    
    # Determine subsets to run
    subsets = ["shopping", "admin", "gitlab", "reddit", "multi"] if args.subset == "all" or args.full else [args.subset]
    
    # Determine modes to run
    modes = ["no_memory", "reasoningbank"] if args.mode == "both" or args.full else [args.mode]
    
    logger.info(f"\nRunning experiment:")
    logger.info(f"  Subsets: {subsets}")
    logger.info(f"  Modes: {modes}")
    logger.info(f"  Total evaluations: {len(subsets) * len(modes)}")
    
    # Run evaluations
    success_count = 0
    total_count = 0
    
    for mode in modes:
        for subset in subsets:
            total_count += 1
            if run_evaluation(mode, subset, args.model, args.seed):
                success_count += 1
    
    logger.info(f"\nCompleted {success_count}/{total_count} evaluations successfully")
    
    # Generate Table 1 if we ran both modes
    if "no_memory" in modes and "reasoningbank" in modes:
        generate_table1(args.model)
    
    logger.info("\nExperiment complete!")
    logger.info("To generate final table, run:")
    logger.info("  python aggregate_results.py")


if __name__ == "__main__":
    main()
