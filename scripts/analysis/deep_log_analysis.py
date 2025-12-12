#!/usr/bin/env python3
"""
Deep Log Analysis - Extract Actionable Insights
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path("/Users/hivamoh/Desktop/ReasoningBank")

def analyze_error_patterns():
    """Extract and categorize all error patterns"""
    print("\n" + "="*70)
    print("DEEP ERROR PATTERN ANALYSIS")
    print("="*70)

    error_categories = {
        'loop_detection': [],
        'parsing_errors': [],
        'connection_errors': [],
        'timeout_errors': [],
        'llm_errors': [],
        'environment_errors': [],
        'other': []
    }

    # Analyze WebArena logs
    log_dirs = [
        BASE_DIR / "logs-good",
        BASE_DIR / "logs_together",
        BASE_DIR / "logs_pro"
    ]

    for log_dir in log_dirs:
        if not log_dir.exists():
            continue

        for log_file in log_dir.glob("*.log"):
            try:
                content = log_file.read_text(errors='ignore')

                # Extract all errors
                error_pattern = r'\d{4}-\d{2}-\d{2}.*?\| ERROR\s+\|\s+(.+?)(?=\n\d{4}|\Z)'
                errors = re.findall(error_pattern, content, re.DOTALL)

                for error in errors:
                    error_lower = error.lower()

                    if 'loop detected' in error_lower or 'persistent loop' in error_lower:
                        error_categories['loop_detection'].append(error[:200])
                    elif 'parse' in error_lower or 'json' in error_lower or 'invalid syntax' in error_lower:
                        error_categories['parsing_errors'].append(error[:200])
                    elif 'connection' in error_lower or 'timeout' in error_lower or 'disconnect' in error_lower:
                        error_categories['connection_errors'].append(error[:200])
                    elif 'llm' in error_lower or 'model' in error_lower or 'response' in error_lower:
                        error_categories['llm_errors'].append(error[:200])
                    elif 'environment' in error_lower or 'browsergym' in error_lower or 'playwright' in error_lower:
                        error_categories['environment_errors'].append(error[:200])
                    else:
                        error_categories['other'].append(error[:200])

            except Exception as e:
                print(f"  ERROR reading {log_file.name}: {e}")

    # Print results
    for category, errors in error_categories.items():
        if errors:
            print(f"\n{category.upper().replace('_', ' ')} ({len(errors)} occurrences):")
            # Show unique errors
            unique_errors = set(errors)
            for i, error in enumerate(list(unique_errors)[:5], 1):
                print(f"  {i}. {error.strip()[:150]}...")

    return error_categories

def analyze_performance_from_results():
    """Analyze performance from results JSON files"""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS FROM RESULTS")
    print("="*70)

    results_dirs = [
        (BASE_DIR / "results 28", "Gemini-2.5-Flash"),
        (BASE_DIR / "results_pro", "Gemini-2.5-Pro"),
        (BASE_DIR / "results_together", "Qwen-7B"),
    ]

    all_stats = {}

    for results_dir, model_name in results_dirs:
        if not results_dir.exists():
            continue

        print(f"\n{model_name}:")
        print("-" * 60)

        # WebArena Multi
        multi_nomem = results_dir / "no_memory_multi.json"
        multi_rbank = results_dir / "reasoningbank_multi.json"

        if multi_nomem.exists():
            data = json.loads(multi_nomem.read_text())
            sr = data.get('success_rate', 0) * 100
            steps = data.get('avg_steps', 0)
            tasks = data.get('task_results', [])
            successes = sum(1 for t in tasks if t.get('success', False))
            print(f"  Multi No Memory: {sr:.1f}% SR ({successes}/{len(tasks)}), {steps:.1f} avg steps")

        if multi_rbank.exists():
            data = json.loads(multi_rbank.read_text())
            sr = data.get('success_rate', 0) * 100
            steps = data.get('avg_steps', 0)
            tasks = data.get('task_results', [])
            successes = sum(1 for t in tasks if t.get('success', False))
            print(f"  Multi ReasoningBank: {sr:.1f}% SR ({successes}/{len(tasks)}), {steps:.1f} avg steps")

        # WebArena Reddit
        reddit_nomem = results_dir / "no_memory_reddit.json"
        reddit_rbank = results_dir / "reasoningbank_reddit.json"

        if reddit_nomem.exists():
            data = json.loads(reddit_nomem.read_text())
            sr = data.get('success_rate', 0) * 100
            steps = data.get('avg_steps', 0)
            tasks = data.get('task_results', [])
            successes = sum(1 for t in tasks if t.get('success', False))
            print(f"  Reddit No Memory: {sr:.1f}% SR ({successes}/{len(tasks)}), {steps:.1f} avg steps")

        if reddit_rbank.exists():
            data = json.loads(reddit_rbank.read_text())
            sr = data.get('success_rate', 0) * 100
            steps = data.get('avg_steps', 0)
            tasks = data.get('task_results', [])
            successes = sum(1 for t in tasks if t.get('success', False))
            print(f"  Reddit ReasoningBank: {sr:.1f}% SR ({successes}/{len(tasks)}), {steps:.1f} avg steps")

        # SWE-Bench
        sweb_nomem = results_dir / "swebench_no_memory/results.json"
        sweb_rbank = results_dir / "swebench_reasoningbank/results.json"

        if sweb_nomem.exists():
            data = json.loads(sweb_nomem.read_text())
            if isinstance(data, list):
                successes = sum(1 for t in data if t.get('success', False))
                total = len(data)
                sr = successes / total * 100 if total > 0 else 0
                print(f"  SWE-Bench No Memory: {sr:.1f}% SR ({successes}/{total})")

        if sweb_rbank.exists():
            data = json.loads(sweb_rbank.read_text())
            if isinstance(data, list):
                successes = sum(1 for t in data if t.get('success', False))
                total = len(data)
                sr = successes / total * 100 if total > 0 else 0
                print(f"  SWE-Bench ReasoningBank: {sr:.1f}% SR ({successes}/{total})")

def analyze_failure_modes():
    """Analyze common failure modes from task results"""
    print("\n" + "="*70)
    print("FAILURE MODE ANALYSIS")
    print("="*70)

    failure_reasons = Counter()
    success_patterns = Counter()

    # Analyze task results
    results_files = list(Path(BASE_DIR / "results 28").glob("*.json"))

    for result_file in results_files:
        try:
            data = json.loads(result_file.read_text())

            if 'task_results' in data:
                tasks = data['task_results']
            elif isinstance(data, list):
                tasks = data
            else:
                continue

            for task in tasks:
                if task.get('success', False):
                    # Analyze success
                    steps = task.get('steps', 0)
                    if steps < 5:
                        success_patterns['quick_success_<5_steps'] += 1
                    elif steps < 15:
                        success_patterns['medium_success_5-15_steps'] += 1
                    else:
                        success_patterns['long_success_15+_steps'] += 1
                else:
                    # Analyze failure
                    error = task.get('error', '')
                    steps = task.get('steps', 0)

                    if error:
                        if 'timeout' in error.lower():
                            failure_reasons['timeout'] += 1
                        elif 'loop' in error.lower():
                            failure_reasons['stuck_in_loop'] += 1
                        elif 'parse' in error.lower():
                            failure_reasons['parse_error'] += 1
                        elif 'connection' in error.lower():
                            failure_reasons['connection_error'] += 1
                        else:
                            failure_reasons['other_error'] += 1
                    elif steps >= 30:
                        failure_reasons['max_steps_reached'] += 1
                    else:
                        failure_reasons['task_failed_unknown'] += 1

        except Exception as e:
            print(f"  ERROR reading {result_file.name}: {e}")

    print("\nTop Failure Reasons:")
    for reason, count in failure_reasons.most_common(10):
        print(f"  {reason}: {count}")

    print("\nSuccess Patterns:")
    for pattern, count in success_patterns.most_common():
        print(f"  {pattern}: {count}")

def generate_actionable_insights(error_categories):
    """Generate actionable recommendations"""
    print("\n" + "="*70)
    print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("="*70)

    insights = []

    # Loop detection insights
    loop_count = len(error_categories['loop_detection'])
    if loop_count > 10:
        insights.append(f"""
1. LOOP DETECTION ({loop_count} occurrences)
   Problem: Agent gets stuck in repetitive action loops
   Root Cause: Lack of state awareness or memory of recent actions
   Recommendation:
   - Implement action history tracking (last 5 actions)
   - Add loop detection with early exit
   - Use memory to store "failed attempts" to avoid repeating
   - Consider adding state fingerprinting
        """)

    # Parsing errors
    parse_count = len(error_categories['parsing_errors'])
    if parse_count > 5:
        insights.append(f"""
2. PARSING ERRORS ({parse_count} occurrences)
   Problem: LLM generates malformed actions or invalid JSON
   Root Cause: Prompt engineering or model hallucination
   Recommendation:
   - Add structured output validation
   - Implement retry logic with error feedback
   - Use few-shot examples in prompts
   - Consider fine-tuning or better prompt templates
        """)

    # Connection/timeout errors
    conn_count = len(error_categories['connection_errors'])
    if conn_count > 5:
        insights.append(f"""
3. CONNECTION/TIMEOUT ERRORS ({conn_count} occurrences)
   Problem: Browser/environment connection issues
   Root Cause: AWS EC2 latency or WebArena container issues
   Recommendation:
   - Increase timeout values
   - Add retry logic for transient failures
   - Monitor EC2 instance health
   - Consider using local WebArena instances for development
        """)

    # Environment errors
    env_count = len(error_categories['environment_errors'])
    if env_count > 5:
        insights.append(f"""
4. ENVIRONMENT ERRORS ({env_count} occurrences)
   Problem: BrowserGym/Playwright setup issues
   Root Cause: Version mismatches or configuration problems
   Recommendation:
   - Pin exact versions of BrowserGym and Playwright
   - Add environment health checks before each run
   - Document exact setup requirements
   - Consider Docker containerization for reproducibility
        """)

    # General recommendations
    insights.append("""
5. GENERAL RECOMMENDATIONS
   a) Monitoring & Logging:
      - Add structured logging with task IDs
      - Track memory retrieval hit rate
      - Monitor token usage per task
      - Log action sequences for failure analysis

   b) Memory System Improvements:
      - Implement memory quality scoring
      - Add memory consolidation (merge similar memories)
      - Filter low-quality memories (quality < 0.3)
      - Track memory usage statistics

   c) Evaluation Improvements:
      - Run each task 3x with different seeds
      - Add statistical significance tests
      - Track variance across runs
      - Report confidence intervals
    """)

    for insight in insights:
        print(insight)

def main():
    """Run deep log analysis"""
    print("="*70)
    print("DEEP LOG ANALYSIS - ACTIONABLE INSIGHTS")
    print("="*70)

    # 1. Error pattern analysis
    error_categories = analyze_error_patterns()

    # 2. Performance analysis
    analyze_performance_from_results()

    # 3. Failure mode analysis
    analyze_failure_modes()

    # 4. Generate actionable insights
    generate_actionable_insights(error_categories)

    # Save summary
    summary_path = BASE_DIR / "DEEP_ANALYSIS_INSIGHTS.txt"
    print(f"\n✅ Full analysis saved to: {summary_path}")

if __name__ == "__main__":
    main()
