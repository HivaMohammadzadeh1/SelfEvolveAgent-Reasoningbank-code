"""
Comprehensive Analysis of ReasoningBank Results for Paper
This script generates detailed analysis including:
- Distribution analysis by subset
- Hard vs easy task analysis
- Performance by category
- Ablation studies
- Memory metrics (retrieval quality, content, quantity)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

# Paths
RESULTS_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/results copy')
LOGS_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/logs copy')
MEMORY_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/memory_bank_swebench_reasoningbank copy')
OUTPUT_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/paper_analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("REASONINGBANK COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
print("\n1. Loading data...")

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load multi-turn dialogue results
no_memory_multi = load_json(RESULTS_DIR / 'no_memory_multi.json')
reasoningbank_multi = load_json(RESULTS_DIR / 'reasoningbank_multi.json')

# Load SWEBench results
swebench_no_memory_summary = load_json(RESULTS_DIR / 'swebench_no_memory' / 'summary.json')
swebench_reasoningbank_summary = load_json(RESULTS_DIR / 'swebench_reasoningbank' / 'summary.json')

# Load memory bank
memories = load_jsonl(MEMORY_DIR / 'memories.jsonl')

print(f"✓ Loaded multi-turn dialogue results")
print(f"  - No memory: {len(no_memory_multi['task_results'])} tasks")
print(f"  - ReasoningBank: {len(reasoningbank_multi['task_results'])} tasks")
print(f"✓ Loaded SWEBench results")
print(f"  - No memory: {len(swebench_no_memory_summary)} tasks")
print(f"  - ReasoningBank: {len(swebench_reasoningbank_summary)} tasks")
print(f"✓ Loaded {len(memories)} memories from ReasoningBank")

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n2. Computing basic statistics...")

def compute_stats(results):
    """Compute statistics for a set of results"""
    tasks = results if isinstance(results, list) else results.get('task_results', [])

    # Filter out tasks with errors (network issues, etc.)
    valid_tasks = [t for t in tasks if t.get('error') is None or 'ERR_' not in str(t.get('error', ''))]
    successful_tasks = [t for t in valid_tasks if t.get('success', False)]

    stats = {
        'total_tasks': len(tasks),
        'valid_tasks': len(valid_tasks),
        'successful_tasks': len(successful_tasks),
        'success_rate': len(successful_tasks) / len(valid_tasks) if valid_tasks else 0,
        'avg_steps': np.mean([t['steps'] for t in valid_tasks]) if valid_tasks else 0,
        'avg_tokens': np.mean([t.get('tokens_input', 0) + t.get('tokens_output', 0) for t in valid_tasks]) if valid_tasks else 0,
        'avg_walltime': np.mean([t.get('walltime', 0) for t in valid_tasks]) if valid_tasks else 0,
    }
    return stats, valid_tasks, successful_tasks

multi_no_mem_stats, multi_no_mem_valid, multi_no_mem_success = compute_stats(no_memory_multi)
multi_rb_stats, multi_rb_valid, multi_rb_success = compute_stats(reasoningbank_multi)
swe_no_mem_stats, swe_no_mem_valid, swe_no_mem_success = compute_stats(swebench_no_memory_summary)
swe_rb_stats, swe_rb_valid, swe_rb_success = compute_stats(swebench_reasoningbank_summary)

# Create summary table
summary_df = pd.DataFrame({
    'Multi-Turn (No Memory)': multi_no_mem_stats,
    'Multi-Turn (ReasoningBank)': multi_rb_stats,
    'SWEBench (No Memory)': swe_no_mem_stats,
    'SWEBench (ReasoningBank)': swe_rb_stats,
}).T

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(summary_df.to_string())

# Save summary
summary_df.to_csv(OUTPUT_DIR / 'summary_statistics.csv')
summary_df.to_latex(OUTPUT_DIR / 'summary_statistics.tex', float_format="%.3f")

# ============================================================================
# 3. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n3. Analyzing distributions...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Steps distribution
ax = axes[0, 0]
steps_data = {
    'No Memory': [t['steps'] for t in multi_no_mem_valid + swe_no_mem_valid if t['steps'] > 0],
    'ReasoningBank': [t['steps'] for t in multi_rb_valid + swe_rb_valid if t['steps'] > 0]
}
ax.hist([steps_data['No Memory'], steps_data['ReasoningBank']],
        bins=20, label=['No Memory', 'ReasoningBank'], alpha=0.7)
ax.set_xlabel('Number of Steps')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Steps per Task')
ax.legend()
ax.grid(True, alpha=0.3)

# Tokens distribution
ax = axes[0, 1]
tokens_data = {
    'No Memory': [t.get('tokens_input', 0) + t.get('tokens_output', 0)
                  for t in multi_no_mem_valid + swe_no_mem_valid
                  if t.get('tokens_input', 0) + t.get('tokens_output', 0) > 0],
    'ReasoningBank': [t.get('tokens_input', 0) + t.get('tokens_output', 0)
                      for t in multi_rb_valid + swe_rb_valid
                      if t.get('tokens_input', 0) + t.get('tokens_output', 0) > 0]
}
ax.hist([tokens_data['No Memory'], tokens_data['ReasoningBank']],
        bins=20, label=['No Memory', 'ReasoningBank'], alpha=0.7)
ax.set_xlabel('Total Tokens')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Tokens per Task')
ax.legend()
ax.grid(True, alpha=0.3)

# Walltime distribution
ax = axes[1, 0]
walltime_data = {
    'No Memory': [t.get('walltime', 0) for t in multi_no_mem_valid + swe_no_mem_valid if t.get('walltime', 0) > 0],
    'ReasoningBank': [t.get('walltime', 0) for t in multi_rb_valid + swe_rb_valid if t.get('walltime', 0) > 0]
}
ax.hist([walltime_data['No Memory'], walltime_data['ReasoningBank']],
        bins=20, label=['No Memory', 'ReasoningBank'], alpha=0.7)
ax.set_xlabel('Walltime (seconds)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Walltime per Task')
ax.legend()
ax.grid(True, alpha=0.3)

# Success rate by dataset
ax = axes[1, 1]
datasets = ['Multi-Turn', 'SWEBench']
no_mem_sr = [multi_no_mem_stats['success_rate'], swe_no_mem_stats['success_rate']]
rb_sr = [multi_rb_stats['success_rate'], swe_rb_stats['success_rate']]

x = np.arange(len(datasets))
width = 0.35

ax.bar(x - width/2, no_mem_sr, width, label='No Memory', alpha=0.8)
ax.bar(x + width/2, rb_sr, width, label='ReasoningBank', alpha=0.8)
ax.set_ylabel('Success Rate')
ax.set_title('Success Rate by Dataset')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, max(no_mem_sr + rb_sr) * 1.2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'distributions.pdf', bbox_inches='tight')
print(f"✓ Saved distribution plots")

# ============================================================================
# 4. HARD VS EASY TASK ANALYSIS
# ============================================================================
print("\n4. Analyzing hard vs easy tasks...")

def categorize_difficulty(task):
    """Categorize task difficulty based on steps and success"""
    steps = task.get('steps', 0)
    success = task.get('success', False)

    # Define difficulty criteria
    if success and steps <= 10:
        return 'Easy'
    elif success and 10 < steps <= 20:
        return 'Medium'
    elif success and steps > 20:
        return 'Hard (Solved)'
    elif not success and steps < 20:
        return 'Hard (Failed, Few Steps)'
    else:
        return 'Hard (Failed, Many Steps)'

# Categorize all tasks
all_tasks_no_mem = multi_no_mem_valid + swe_no_mem_valid
all_tasks_rb = multi_rb_valid + swe_rb_valid

difficulty_no_mem = Counter([categorize_difficulty(t) for t in all_tasks_no_mem])
difficulty_rb = Counter([categorize_difficulty(t) for t in all_tasks_rb])

difficulty_df = pd.DataFrame({
    'No Memory': difficulty_no_mem,
    'ReasoningBank': difficulty_rb
}).fillna(0).T

print("\n" + "="*80)
print("TASK DIFFICULTY DISTRIBUTION")
print("="*80)
print(difficulty_df.to_string())
difficulty_df.to_csv(OUTPUT_DIR / 'difficulty_distribution.csv')

# Plot difficulty distribution
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
difficulty_df.T.plot(kind='bar', ax=ax, alpha=0.8)
ax.set_xlabel('Difficulty Category')
ax.set_ylabel('Number of Tasks')
ax.set_title('Task Difficulty Distribution: No Memory vs ReasoningBank')
ax.legend(title='Method')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'difficulty_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'difficulty_distribution.pdf', bbox_inches='tight')
print(f"✓ Saved difficulty distribution plot")

# ============================================================================
# 5. PERFORMANCE BY CATEGORY
# ============================================================================
print("\n5. Analyzing performance by category...")

# Analyze by benchmark type
benchmark_stats = pd.DataFrame({
    'Multi-Turn': {
        'No Memory Success Rate': multi_no_mem_stats['success_rate'],
        'ReasoningBank Success Rate': multi_rb_stats['success_rate'],
        'Improvement': (multi_rb_stats['success_rate'] - multi_no_mem_stats['success_rate']) / multi_no_mem_stats['success_rate'] * 100 if multi_no_mem_stats['success_rate'] > 0 else 0,
        'No Memory Avg Steps': multi_no_mem_stats['avg_steps'],
        'ReasoningBank Avg Steps': multi_rb_stats['avg_steps'],
        'Steps Change %': (multi_rb_stats['avg_steps'] - multi_no_mem_stats['avg_steps']) / multi_no_mem_stats['avg_steps'] * 100 if multi_no_mem_stats['avg_steps'] > 0 else 0,
    },
    'SWEBench': {
        'No Memory Success Rate': swe_no_mem_stats['success_rate'],
        'ReasoningBank Success Rate': swe_rb_stats['success_rate'],
        'Improvement': (swe_rb_stats['success_rate'] - swe_no_mem_stats['success_rate']) / swe_no_mem_stats['success_rate'] * 100 if swe_no_mem_stats['success_rate'] > 0 else 0,
        'No Memory Avg Steps': swe_no_mem_stats['avg_steps'],
        'ReasoningBank Avg Steps': swe_rb_stats['avg_steps'],
        'Steps Change %': (swe_rb_stats['avg_steps'] - swe_no_mem_stats['avg_steps']) / swe_no_mem_stats['avg_steps'] * 100 if swe_no_mem_stats['avg_steps'] > 0 else 0,
    }
}).T

print("\n" + "="*80)
print("PERFORMANCE BY BENCHMARK")
print("="*80)
print(benchmark_stats.to_string())
benchmark_stats.to_csv(OUTPUT_DIR / 'benchmark_performance.csv')
benchmark_stats.to_latex(OUTPUT_DIR / 'benchmark_performance.tex', float_format="%.2f")

# ============================================================================
# 6. MEMORY ANALYSIS
# ============================================================================
print("\n6. Analyzing memory bank...")

# Analyze memory provenance
memory_success = [m for m in memories if m['provenance']['success']]
memory_failure = [m for m in memories if not m['provenance']['success']]

memory_stats = {
    'Total Memories': len(memories),
    'From Successful Tasks': len(memory_success),
    'From Failed Tasks': len(memory_failure),
    'Success Rate': len(memory_success) / len(memories) if memories else 0,
    'Avg Steps (Successful)': np.mean([m['provenance']['steps'] for m in memory_success]) if memory_success else 0,
    'Avg Steps (Failed)': np.mean([m['provenance']['steps'] for m in memory_failure]) if memory_failure else 0,
}

print("\n" + "="*80)
print("MEMORY BANK STATISTICS")
print("="*80)
for key, value in memory_stats.items():
    print(f"{key:30s}: {value}")

# Analyze memory content
memory_titles = Counter([m['title'] for m in memories])
memory_descriptions = Counter([m['description'] for m in memories])

print("\n" + "="*80)
print("TOP 10 MEMORY TYPES (by title)")
print("="*80)
for title, count in memory_titles.most_common(10):
    print(f"{count:4d} | {title}")

# Plot memory type distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top memory types
ax = axes[0]
top_memories = dict(memory_titles.most_common(15))
ax.barh(list(top_memories.keys()), list(top_memories.values()), alpha=0.8)
ax.set_xlabel('Count')
ax.set_title('Top 15 Memory Types in ReasoningBank')
ax.grid(True, alpha=0.3, axis='x')
plt.setp(ax.get_yticklabels(), fontsize=9)

# Memory success rate
ax = axes[1]
categories = ['Successful Tasks', 'Failed Tasks']
counts = [len(memory_success), len(memory_failure)]
colors = ['#2ecc71', '#e74c3c']
ax.bar(categories, counts, color=colors, alpha=0.8)
ax.set_ylabel('Number of Memories')
ax.set_title('Memory Bank: Source Task Success')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(counts):
    ax.text(i, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'memory_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'memory_analysis.pdf', bbox_inches='tight')
print(f"✓ Saved memory analysis plots")

# Save memory stats
with open(OUTPUT_DIR / 'memory_statistics.json', 'w') as f:
    json.dump(memory_stats, f, indent=2)

# ============================================================================
# 7. ABLATION STUDY: MEMORY QUANTITY VS PERFORMANCE
# ============================================================================
print("\n7. Performing ablation analysis...")

# Analyze relationship between memory size and performance
# Group tasks by whether they succeeded, and look at steps
rb_success_steps = [t['steps'] for t in all_tasks_rb if t.get('success', False) and t['steps'] > 0]
rb_failure_steps = [t['steps'] for t in all_tasks_rb if not t.get('success', False) and t['steps'] > 0]
nm_success_steps = [t['steps'] for t in all_tasks_no_mem if t.get('success', False) and t['steps'] > 0]
nm_failure_steps = [t['steps'] for t in all_tasks_no_mem if not t.get('success', False) and t['steps'] > 0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot comparison
ax = axes[0]
data_to_plot = [nm_success_steps, nm_failure_steps, rb_success_steps, rb_failure_steps]
labels = ['NM Success', 'NM Failure', 'RB Success', 'RB Failure']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Number of Steps')
ax.set_title('Steps Distribution: Success vs Failure')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Efficiency comparison
ax = axes[1]
efficiency_data = {
    'Method': ['No Memory', 'ReasoningBank'],
    'Success Rate': [
        (multi_no_mem_stats['success_rate'] + swe_no_mem_stats['success_rate']) / 2,
        (multi_rb_stats['success_rate'] + swe_rb_stats['success_rate']) / 2
    ],
    'Avg Steps': [
        (multi_no_mem_stats['avg_steps'] + swe_no_mem_stats['avg_steps']) / 2,
        (multi_rb_stats['avg_steps'] + swe_rb_stats['avg_steps']) / 2
    ]
}

ax2 = ax.twinx()
x = np.arange(len(efficiency_data['Method']))
width = 0.35

bars1 = ax.bar(x - width/2, efficiency_data['Success Rate'], width,
               label='Success Rate', alpha=0.8, color='#2ecc71')
bars2 = ax2.bar(x + width/2, efficiency_data['Avg Steps'], width,
                label='Avg Steps', alpha=0.8, color='#3498db')

ax.set_ylabel('Success Rate', color='#2ecc71')
ax2.set_ylabel('Average Steps', color='#3498db')
ax.set_title('Efficiency: Success Rate vs Steps')
ax.set_xticks(x)
ax.set_xticklabels(efficiency_data['Method'])
ax.tick_params(axis='y', labelcolor='#2ecc71')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        (ax if bars == bars1 else ax2).text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ablation_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'ablation_analysis.pdf', bbox_inches='tight')
print(f"✓ Saved ablation analysis plots")

# ============================================================================
# 8. DETAILED METRICS COMPARISON
# ============================================================================
print("\n8. Computing detailed metrics...")

detailed_metrics = pd.DataFrame({
    'Metric': [
        'Success Rate (%)',
        'Avg Steps',
        'Avg Tokens (K)',
        'Avg Walltime (min)',
        'Total Tasks',
        'Valid Tasks',
        'Successful Tasks',
    ],
    'Multi-Turn\nNo Memory': [
        multi_no_mem_stats['success_rate'] * 100,
        multi_no_mem_stats['avg_steps'],
        multi_no_mem_stats['avg_tokens'] / 1000,
        multi_no_mem_stats['avg_walltime'] / 60,
        multi_no_mem_stats['total_tasks'],
        multi_no_mem_stats['valid_tasks'],
        multi_no_mem_stats['successful_tasks'],
    ],
    'Multi-Turn\nReasoningBank': [
        multi_rb_stats['success_rate'] * 100,
        multi_rb_stats['avg_steps'],
        multi_rb_stats['avg_tokens'] / 1000,
        multi_rb_stats['avg_walltime'] / 60,
        multi_rb_stats['total_tasks'],
        multi_rb_stats['valid_tasks'],
        multi_rb_stats['successful_tasks'],
    ],
    'SWEBench\nNo Memory': [
        swe_no_mem_stats['success_rate'] * 100,
        swe_no_mem_stats['avg_steps'],
        swe_no_mem_stats['avg_tokens'] / 1000,
        swe_no_mem_stats['avg_walltime'] / 60,
        swe_no_mem_stats['total_tasks'],
        swe_no_mem_stats['valid_tasks'],
        swe_no_mem_stats['successful_tasks'],
    ],
    'SWEBench\nReasoningBank': [
        swe_rb_stats['success_rate'] * 100,
        swe_rb_stats['avg_steps'],
        swe_rb_stats['avg_tokens'] / 1000,
        swe_rb_stats['avg_walltime'] / 60,
        swe_rb_stats['total_tasks'],
        swe_rb_stats['valid_tasks'],
        swe_rb_stats['successful_tasks'],
    ],
})

print("\n" + "="*80)
print("DETAILED METRICS COMPARISON")
print("="*80)
print(detailed_metrics.to_string(index=False))
detailed_metrics.to_csv(OUTPUT_DIR / 'detailed_metrics.csv', index=False)
detailed_metrics.to_latex(OUTPUT_DIR / 'detailed_metrics.tex', index=False, float_format="%.2f")

# ============================================================================
# 9. GENERATE LATEX TABLES FOR PAPER
# ============================================================================
print("\n9. Generating LaTeX tables...")

# Main results table
main_results = pd.DataFrame({
    'Benchmark': ['Multi-Turn Dialogue', 'SWEBench'],
    'No Memory SR (%)': [
        f"{multi_no_mem_stats['success_rate']*100:.1f}",
        f"{swe_no_mem_stats['success_rate']*100:.1f}",
    ],
    'ReasoningBank SR (%)': [
        f"{multi_rb_stats['success_rate']*100:.1f}",
        f"{swe_rb_stats['success_rate']*100:.1f}",
    ],
    'Improvement': [
        f"+{((multi_rb_stats['success_rate'] - multi_no_mem_stats['success_rate']) / multi_no_mem_stats['success_rate'] * 100):.1f}%" if multi_no_mem_stats['success_rate'] > 0 else "N/A",
        f"+{((swe_rb_stats['success_rate'] - swe_no_mem_stats['success_rate']) / swe_no_mem_stats['success_rate'] * 100):.1f}%" if swe_no_mem_stats['success_rate'] > 0 else "N/A",
    ],
    'Avg Steps (NM)': [
        f"{multi_no_mem_stats['avg_steps']:.1f}",
        f"{swe_no_mem_stats['avg_steps']:.1f}",
    ],
    'Avg Steps (RB)': [
        f"{multi_rb_stats['avg_steps']:.1f}",
        f"{swe_rb_stats['avg_steps']:.1f}",
    ],
})

with open(OUTPUT_DIR / 'main_results_table.tex', 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Main Results: ReasoningBank vs No Memory Baseline}\n")
    f.write("\\label{tab:main_results}\n")
    f.write(main_results.to_latex(index=False, escape=False))
    f.write("\\end{table}\n")

print(f"✓ Generated LaTeX tables")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  • summary_statistics.csv/tex")
print("  • distributions.png/pdf")
print("  • difficulty_distribution.csv/png/pdf")
print("  • benchmark_performance.csv/tex")
print("  • memory_analysis.png/pdf")
print("  • memory_statistics.json")
print("  • ablation_analysis.png/pdf")
print("  • detailed_metrics.csv/tex")
print("  • main_results_table.tex")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
overall_improvement = ((multi_rb_stats['success_rate'] + swe_rb_stats['success_rate']) / 2 -
                       (multi_no_mem_stats['success_rate'] + swe_no_mem_stats['success_rate']) / 2) / \
                      ((multi_no_mem_stats['success_rate'] + swe_no_mem_stats['success_rate']) / 2) * 100

print(f"1. Overall Success Rate Improvement: {overall_improvement:.1f}%")
print(f"2. Memory Bank Size: {len(memories)} stored strategies")
print(f"3. Memory Success Rate: {memory_stats['Success Rate']*100:.1f}%")
print(f"4. Multi-Turn Improvement: {((multi_rb_stats['success_rate'] - multi_no_mem_stats['success_rate']) / multi_no_mem_stats['success_rate'] * 100):.1f}%")
print(f"5. SWEBench Improvement: {((swe_rb_stats['success_rate'] - swe_no_mem_stats['success_rate']) / swe_no_mem_stats['success_rate'] * 100):.1f}%")
print("="*80)
