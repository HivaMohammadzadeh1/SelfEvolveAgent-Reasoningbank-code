"""
Advanced Analysis for ReasoningBank
Focus on:
- Retrieval quality metrics
- Memory content and strategy analysis
- Performance correlation with memory usage
- Task-specific analysis
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11

# Paths
RESULTS_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/results copy')
LOGS_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/logs copy')
MEMORY_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/memory_bank_swebench_reasoningbank copy')
OUTPUT_DIR = Path('/Users/hivamoh/Desktop/ReasoningBank/paper_analysis')

print("="*80)
print("ADVANCED REASONINGBANK ANALYSIS")
print("="*80)

# Load data
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

memories = load_jsonl(MEMORY_DIR / 'memories.jsonl')
swebench_no_memory_data = load_json(RESULTS_DIR / 'swebench_no_memory' / 'summary.json')
swebench_reasoningbank_data = load_json(RESULTS_DIR / 'swebench_reasoningbank' / 'summary.json')

swebench_no_memory = swebench_no_memory_data['task_results']
swebench_reasoningbank = swebench_reasoningbank_data['task_results']

print(f"\n✓ Loaded {len(memories)} memories")
print(f"✓ Loaded {len(swebench_no_memory)} SWEBench no-memory results")
print(f"✓ Loaded {len(swebench_reasoningbank)} SWEBench reasoningbank results")

# ============================================================================
# 1. MEMORY CONTENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. MEMORY CONTENT ANALYSIS")
print("="*80)

# Extract strategy types and content
strategy_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'avg_steps': []})

for mem in memories:
    title = mem['title']
    success = mem['provenance']['success']
    steps = mem['provenance']['steps']

    strategy_stats[title]['count'] += 1
    if success:
        strategy_stats[title]['success'] += 1
    strategy_stats[title]['avg_steps'].append(steps)

# Create detailed strategy dataframe
strategy_df = pd.DataFrame([
    {
        'Strategy': title,
        'Count': stats['count'],
        'Success_Count': stats['success'],
        'Success_Rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0,
        'Avg_Steps': np.mean(stats['avg_steps']) if stats['avg_steps'] else 0,
        'Std_Steps': np.std(stats['avg_steps']) if stats['avg_steps'] else 0,
    }
    for title, stats in strategy_stats.items()
]).sort_values('Count', ascending=False)

print("\nTop 20 Strategies by Frequency:")
print(strategy_df.head(20).to_string(index=False))

strategy_df.to_csv(OUTPUT_DIR / 'strategy_details.csv', index=False)

# Plot top strategies
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top 15 strategies by count
ax = axes[0]
top_15 = strategy_df.head(15)
colors = ['#2ecc71' if sr > 0.5 else '#e74c3c' for sr in top_15['Success_Rate']]
ax.barh(range(len(top_15)), top_15['Count'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['Strategy'], fontsize=9)
ax.set_xlabel('Frequency in Memory Bank')
ax.set_title('Top 15 Strategies by Frequency (Green = High Success Rate)')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Success rate vs frequency
ax = axes[1]
scatter_data = strategy_df[strategy_df['Count'] >= 2]  # Only strategies with 2+ occurrences
sizes = scatter_data['Count'] * 50
colors_scatter = scatter_data['Success_Rate']
scatter = ax.scatter(scatter_data['Count'], scatter_data['Success_Rate'],
                    s=sizes, c=colors_scatter, cmap='RdYlGn', alpha=0.6, edgecolors='black')
ax.set_xlabel('Strategy Frequency')
ax.set_ylabel('Success Rate')
ax.set_title('Strategy Effectiveness: Frequency vs Success Rate (size = frequency)')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Success Rate')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'strategy_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'strategy_analysis.pdf', bbox_inches='tight')
print(f"\n✓ Saved strategy analysis plots")

# ============================================================================
# 2. MEMORY QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. MEMORY QUALITY ANALYSIS")
print("="*80)

# Analyze memory quality by success and steps
quality_bins = [
    ('High Quality (Success, ≤10 steps)', lambda m: m['provenance']['success'] and m['provenance']['steps'] <= 10),
    ('Medium Quality (Success, 11-20 steps)', lambda m: m['provenance']['success'] and 11 <= m['provenance']['steps'] <= 20),
    ('Low Quality (Success, >20 steps)', lambda m: m['provenance']['success'] and m['provenance']['steps'] > 20),
    ('Failed (≤15 steps)', lambda m: not m['provenance']['success'] and m['provenance']['steps'] <= 15),
    ('Failed (>15 steps)', lambda m: not m['provenance']['success'] and m['provenance']['steps'] > 15),
]

quality_distribution = {name: len([m for m in memories if func(m)])
                       for name, func in quality_bins}

print("\nMemory Quality Distribution:")
for quality, count in quality_distribution.items():
    pct = count / len(memories) * 100
    print(f"  {quality:40s}: {count:4d} ({pct:5.1f}%)")

# Plot quality distribution
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
qualities = list(quality_distribution.keys())
counts = list(quality_distribution.values())
colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']

bars = ax.bar(range(len(qualities)), counts, color=colors, alpha=0.8)
ax.set_xticks(range(len(qualities)))
ax.set_xticklabels(qualities, rotation=45, ha='right')
ax.set_ylabel('Number of Memories')
ax.set_title('Memory Quality Distribution in ReasoningBank')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(memories)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'memory_quality.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'memory_quality.pdf', bbox_inches='tight')
print(f"✓ Saved memory quality plots")

# ============================================================================
# 3. PERFORMANCE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. PERFORMANCE CORRELATION ANALYSIS")
print("="*80)

# Analyze correlation between task characteristics
rb_valid = [t for t in swebench_reasoningbank if t.get('error') is None or 'ERR_' not in str(t.get('error', ''))]
nm_valid = [t for t in swebench_no_memory if t.get('error') is None or 'ERR_' not in str(t.get('error', ''))]

# Create comparison dataframe
comparison_data = []
for rb_task in rb_valid:
    task_id = rb_task.get('task_id')
    nm_task = next((t for t in nm_valid if t.get('task_id') == task_id), None)

    if nm_task and rb_task.get('steps', 0) > 0 and nm_task.get('steps', 0) > 0:
        comparison_data.append({
            'task_id': task_id,
            'rb_success': int(rb_task.get('success', False)),
            'nm_success': int(nm_task.get('success', False)),
            'rb_steps': rb_task.get('steps', 0),
            'nm_steps': nm_task.get('steps', 0),
            'steps_diff': rb_task.get('steps', 0) - nm_task.get('steps', 0),
            'steps_ratio': rb_task.get('steps', 0) / nm_task.get('steps', 0) if nm_task.get('steps', 0) > 0 else 1,
            'rb_tokens': rb_task.get('tokens_input', 0) + rb_task.get('tokens_output', 0),
            'nm_tokens': nm_task.get('tokens_input', 0) + nm_task.get('tokens_output', 0),
            'tokens_diff': (rb_task.get('tokens_input', 0) + rb_task.get('tokens_output', 0)) -
                          (nm_task.get('tokens_input', 0) + nm_task.get('tokens_output', 0)),
        })

comparison_df = pd.DataFrame(comparison_data)

print(f"\nFound {len(comparison_df)} matching tasks for comparison")
print(f"\nTasks where ReasoningBank succeeded but No Memory failed: {len(comparison_df[(comparison_df['rb_success'] == 1) & (comparison_df['nm_success'] == 0)])}")
print(f"Tasks where both succeeded: {len(comparison_df[(comparison_df['rb_success'] == 1) & (comparison_df['nm_success'] == 1)])}")
print(f"Tasks where both failed: {len(comparison_df[(comparison_df['rb_success'] == 0) & (comparison_df['nm_success'] == 0)])}")
print(f"Tasks where No Memory succeeded but ReasoningBank failed: {len(comparison_df[(comparison_df['rb_success'] == 0) & (comparison_df['nm_success'] == 1)])}")

# Statistical analysis
print(f"\nStep Efficiency:")
print(f"  Average steps reduction: {comparison_df['steps_diff'].mean():.2f} steps")
print(f"  Median steps reduction: {comparison_df['steps_diff'].median():.2f} steps")
print(f"  Tasks with fewer steps (RB < NM): {len(comparison_df[comparison_df['steps_diff'] < 0])}")
print(f"  Tasks with more steps (RB > NM): {len(comparison_df[comparison_df['steps_diff'] > 0])}")

# Save comparison data
comparison_df.to_csv(OUTPUT_DIR / 'task_comparison.csv', index=False)

# Plot correlation
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Steps comparison
ax = axes[0, 0]
success_colors = ['red' if s == 0 else 'green' for s in comparison_df['rb_success']]
ax.scatter(comparison_df['nm_steps'], comparison_df['rb_steps'],
          c=success_colors, alpha=0.5, s=50)
max_steps = max(comparison_df['nm_steps'].max(), comparison_df['rb_steps'].max())
ax.plot([0, max_steps], [0, max_steps], 'k--', alpha=0.3, label='Equal steps')
ax.set_xlabel('No Memory Steps')
ax.set_ylabel('ReasoningBank Steps')
ax.set_title('Steps Comparison (Green = RB Success, Red = RB Failure)')
ax.legend()
ax.grid(True, alpha=0.3)

# Steps difference distribution
ax = axes[0, 1]
ax.hist(comparison_df['steps_diff'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.axvline(x=comparison_df['steps_diff'].mean(), color='green', linestyle='--', linewidth=2,
          label=f'Mean: {comparison_df["steps_diff"].mean():.1f}')
ax.set_xlabel('Step Difference (RB - NM)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Step Differences')
ax.legend()
ax.grid(True, alpha=0.3)

# Token comparison
ax = axes[1, 0]
ax.scatter(comparison_df['nm_tokens'], comparison_df['rb_tokens'],
          c=success_colors, alpha=0.5, s=50)
max_tokens = max(comparison_df['nm_tokens'].max(), comparison_df['rb_tokens'].max())
ax.plot([0, max_tokens], [0, max_tokens], 'k--', alpha=0.3, label='Equal tokens')
ax.set_xlabel('No Memory Tokens')
ax.set_ylabel('ReasoningBank Tokens')
ax.set_title('Token Usage Comparison (Green = RB Success, Red = RB Failure)')
ax.legend()
ax.grid(True, alpha=0.3)

# Success rate by step range
ax = axes[1, 1]
step_ranges = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
range_labels = ['0-10', '10-20', '20-30', '30+']

rb_sr_by_range = []
nm_sr_by_range = []

for (low, high), label in zip(step_ranges, range_labels):
    rb_in_range = comparison_df[(comparison_df['rb_steps'] >= low) & (comparison_df['rb_steps'] < high)]
    nm_in_range = comparison_df[(comparison_df['nm_steps'] >= low) & (comparison_df['nm_steps'] < high)]

    rb_sr = rb_in_range['rb_success'].mean() if len(rb_in_range) > 0 else 0
    nm_sr = nm_in_range['nm_success'].mean() if len(nm_in_range) > 0 else 0

    rb_sr_by_range.append(rb_sr)
    nm_sr_by_range.append(nm_sr)

x = np.arange(len(range_labels))
width = 0.35

ax.bar(x - width/2, nm_sr_by_range, width, label='No Memory', alpha=0.8)
ax.bar(x + width/2, rb_sr_by_range, width, label='ReasoningBank', alpha=0.8)
ax.set_ylabel('Success Rate')
ax.set_title('Success Rate by Step Range')
ax.set_xticks(x)
ax.set_xticklabels(range_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'correlation_analysis.pdf', bbox_inches='tight')
print(f"✓ Saved correlation analysis plots")

# ============================================================================
# 4. TASK DIFFICULTY CHARACTERIZATION
# ============================================================================
print("\n" + "="*80)
print("4. TASK DIFFICULTY CHARACTERIZATION")
print("="*80)

# Define difficulty based on baseline performance
def characterize_difficulty(task_id, nm_task, rb_task):
    """Characterize task difficulty"""
    nm_success = nm_task.get('success', False)
    rb_success = rb_task.get('success', False)
    nm_steps = nm_task.get('steps', 0)
    rb_steps = rb_task.get('steps', 0)

    if nm_success and nm_steps <= 15:
        return 'Easy'
    elif nm_success and nm_steps > 15:
        return 'Medium'
    elif not nm_success and rb_success:
        return 'Hard (RB Solved)'
    elif not nm_success and not rb_success and nm_steps < 20:
        return 'Very Hard (Both Failed, Few Steps)'
    else:
        return 'Very Hard (Both Failed, Many Steps)'

difficulty_characterization = []
for rb_task in rb_valid:
    task_id = rb_task.get('task_id')
    nm_task = next((t for t in nm_valid if t.get('task_id') == task_id), None)

    if nm_task:
        diff = characterize_difficulty(task_id, nm_task, rb_task)
        difficulty_characterization.append({
            'task_id': task_id,
            'difficulty': diff,
            'nm_success': nm_task.get('success', False),
            'rb_success': rb_task.get('success', False),
            'nm_steps': nm_task.get('steps', 0),
            'rb_steps': rb_task.get('steps', 0),
        })

difficulty_df = pd.DataFrame(difficulty_characterization)
difficulty_counts = difficulty_df['difficulty'].value_counts()

print("\nTask Difficulty Distribution:")
for diff, count in difficulty_counts.items():
    print(f"  {diff:40s}: {count:4d} ({count/len(difficulty_df)*100:5.1f}%)")

# Calculate RB improvement by difficulty
print("\nReasoningBank Performance by Difficulty:")
for diff in difficulty_counts.index:
    subset = difficulty_df[difficulty_df['difficulty'] == diff]
    rb_sr = subset['rb_success'].mean()
    nm_sr = subset['nm_success'].mean()
    improvement = (rb_sr - nm_sr) / nm_sr * 100 if nm_sr > 0 else 0
    print(f"  {diff:40s}: NM={nm_sr:.1%}, RB={rb_sr:.1%}, Improvement={improvement:+.1f}%")

# Plot difficulty analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Difficulty distribution
ax = axes[0]
difficulty_counts.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
ax.set_xlabel('Number of Tasks')
ax.set_title('Task Distribution by Difficulty')
ax.grid(True, alpha=0.3, axis='x')

# Success rate by difficulty
ax = axes[1]
diff_categories = difficulty_counts.index.tolist()
nm_rates = [difficulty_df[difficulty_df['difficulty'] == d]['nm_success'].mean() for d in diff_categories]
rb_rates = [difficulty_df[difficulty_df['difficulty'] == d]['rb_success'].mean() for d in diff_categories]

x = np.arange(len(diff_categories))
width = 0.35

ax.barh(x - width/2, nm_rates, width, label='No Memory', alpha=0.8)
ax.barh(x + width/2, rb_rates, width, label='ReasoningBank', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(diff_categories, fontsize=9)
ax.set_xlabel('Success Rate')
ax.set_title('Success Rate by Task Difficulty')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'difficulty_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'difficulty_analysis.pdf', bbox_inches='tight')
print(f"✓ Saved difficulty analysis plots")

# Save difficulty characterization
difficulty_df.to_csv(OUTPUT_DIR / 'task_difficulty_characterization.csv', index=False)

# ============================================================================
# 5. MEMORY CONTENT DIVERSITY
# ============================================================================
print("\n" + "="*80)
print("5. MEMORY CONTENT DIVERSITY ANALYSIS")
print("="*80)

# Analyze unique strategies
unique_titles = len(set(m['title'] for m in memories))
unique_descriptions = len(set(m['description'] for m in memories))

print(f"\nMemory Diversity:")
print(f"  Total memories: {len(memories)}")
print(f"  Unique titles: {unique_titles}")
print(f"  Unique descriptions: {unique_descriptions}")
print(f"  Diversity ratio (unique/total): {unique_titles/len(memories):.2%}")

# Analyze content length
content_lengths = [len(' '.join(m['content'])) for m in memories]
print(f"\nMemory Content Statistics:")
print(f"  Avg content length: {np.mean(content_lengths):.1f} chars")
print(f"  Median content length: {np.median(content_lengths):.1f} chars")
print(f"  Min content length: {np.min(content_lengths)} chars")
print(f"  Max content length: {np.max(content_lengths)} chars")

# ============================================================================
# 6. GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("6. GENERATING SUMMARY REPORT")
print("="*80)

report = f"""
REASONINGBANK ADVANCED ANALYSIS SUMMARY
{'='*80}

1. MEMORY BANK COMPOSITION
   - Total memories stored: {len(memories)}
   - Unique strategy types: {unique_titles}
   - Diversity ratio: {unique_titles/len(memories):.1%}
   - High-quality memories (success, ≤10 steps): {quality_distribution.get('High Quality (Success, ≤10 steps)', 0)} ({quality_distribution.get('High Quality (Success, ≤10 steps)', 0)/len(memories)*100:.1f}%)

2. STRATEGY EFFECTIVENESS
   - Most common strategy: {strategy_df.iloc[0]['Strategy']} ({strategy_df.iloc[0]['Count']} occurrences)
   - Highest success rate strategy: {strategy_df.nlargest(1, 'Success_Rate').iloc[0]['Strategy']} ({strategy_df.nlargest(1, 'Success_Rate').iloc[0]['Success_Rate']:.1%})

3. PERFORMANCE IMPACT
   - Tasks with ReasoningBank improvement: {len(comparison_df[(comparison_df['rb_success'] == 1) & (comparison_df['nm_success'] == 0)])}
   - Average step reduction: {comparison_df['steps_diff'].mean():.2f} steps
   - Token overhead: {comparison_df['tokens_diff'].mean():.0f} tokens

4. TASK DIFFICULTY INSIGHTS
   - Easy tasks: {difficulty_counts.get('Easy', 0)} ({difficulty_counts.get('Easy', 0)/len(difficulty_df)*100:.1f}%)
   - Medium tasks: {difficulty_counts.get('Medium', 0)} ({difficulty_counts.get('Medium', 0)/len(difficulty_df)*100:.1f}%)
   - Hard tasks (RB solved): {difficulty_counts.get('Hard (RB Solved)', 0)} ({difficulty_counts.get('Hard (RB Solved)', 0)/len(difficulty_df)*100:.1f}%)
   - Very hard tasks: {difficulty_counts.get('Very Hard (Both Failed, Few Steps)', 0) + difficulty_counts.get('Very Hard (Both Failed, Many Steps)', 0)} ({(difficulty_counts.get('Very Hard (Both Failed, Few Steps)', 0) + difficulty_counts.get('Very Hard (Both Failed, Many Steps)', 0))/len(difficulty_df)*100:.1f}%)

{'='*80}
"""

with open(OUTPUT_DIR / 'advanced_analysis_summary.txt', 'w') as f:
    f.write(report)

print(report)
print(f"✓ Saved advanced analysis summary")

print("\n" + "="*80)
print("ADVANCED ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  • strategy_details.csv")
print("  • strategy_analysis.png/pdf")
print("  • memory_quality.png/pdf")
print("  • task_comparison.csv")
print("  • correlation_analysis.png/pdf")
print("  • task_difficulty_characterization.csv")
print("  • difficulty_analysis.png/pdf")
print("  • advanced_analysis_summary.txt")
print("="*80)
