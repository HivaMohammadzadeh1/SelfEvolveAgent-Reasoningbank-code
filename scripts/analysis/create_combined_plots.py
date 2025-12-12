#!/usr/bin/env python3
"""
Create combined multi-benchmark plots
Combines individual benchmark plots into unified figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from pathlib import Path
from loguru import logger
import numpy as np

# Import dataset loaders
from src.swebench_loader import load_swebench_dataset
from src.webarena_loader import WebArenaDataset
from src.mind2web_loader import load_mind2web_dataset

# Professional publication style
sns.set_style("ticks")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['axes.linewidth'] = 1.2

# Colorblind-friendly palette
COLORS = {
    'easy': '#4477AA',
    'medium': '#CCBB44',
    'hard': '#EE6677',
}


def analyze_all():
    """Analyze all datasets."""
    logger.info("Analyzing all datasets...")

    # SWE-bench
    swebench_dataset = load_swebench_dataset()
    swebench_repo_counts = Counter()
    swebench_difficulties = []

    for task in swebench_dataset.tasks:
        swebench_repo_counts[task.repo] += 1
        patch_lines = len(task.gold_patch.split('\n')) if task.gold_patch else 0
        problem_length = len(task.problem_statement.split())
        difficulty = patch_lines * 0.5 + problem_length * 0.1
        swebench_difficulties.append({'difficulty': difficulty})

    swebench_sorted = sorted(swebench_repo_counts.items(), key=lambda x: x[1], reverse=True)
    swebench_easy_thresh = np.percentile([d['difficulty'] for d in swebench_difficulties], 33)
    swebench_hard_thresh = np.percentile([d['difficulty'] for d in swebench_difficulties], 67)

    # WebArena (exclude 'map' subset, limit multi to 29 - standard 684 tasks as per PRD)
    webarena_dataset = WebArenaDataset(data_dir="data/webarena")
    webarena_tasks = webarena_dataset.load_tasks(
        subsets=["shopping", "admin", "gitlab", "reddit", "multi"],
        max_multi_tasks=29
    )
    webarena_subset_counts = Counter()
    webarena_difficulties = []

    for task in webarena_tasks:
        webarena_subset_counts[task['subset']] += 1

        # More nuanced difficulty scoring for WebArena
        difficulty = 0

        # Multi-domain tasks are harder (cross-site coordination)
        if task['subset'] == 'multi':
            difficulty += 15

        # Login requirement adds complexity
        if task.get('require_login', False):
            difficulty += 3

        # Intent complexity (longer descriptions = more complex tasks)
        intent = task.get('description') or task.get('intent') or ''
        if isinstance(intent, str):
            intent_words = len(intent.split())
            difficulty += min(intent_words * 0.15, 10)  # Cap at 10 points

        # Evaluation complexity (more checks = harder verification)
        eval_spec = task.get('eval') or {}
        if eval_spec:
            # Count evaluation conditions
            num_checks = 0
            if 'program_html' in eval_spec and isinstance(eval_spec['program_html'], list):
                num_checks = len(eval_spec['program_html'])
            elif 'reference_answers' in eval_spec:
                num_checks = len(eval_spec['reference_answers']) if isinstance(eval_spec['reference_answers'], dict) else 1

            difficulty += min(num_checks * 2, 8)  # Cap at 8 points

        webarena_difficulties.append({'difficulty': difficulty, 'subset': task['subset']})

    webarena_sorted = sorted(webarena_subset_counts.items(), key=lambda x: x[1], reverse=True)
    webarena_easy_thresh = np.percentile([d['difficulty'] for d in webarena_difficulties], 33)
    webarena_hard_thresh = np.percentile([d['difficulty'] for d in webarena_difficulties], 67)

    # Mind2Web - Load all three test splits and combine them
    mind2web_domain_counts = Counter()
    mind2web_difficulties = []
    mind2web_total = 0

    for split_name in ["test_task", "test_website", "test_domain"]:
        mind2web_dataset = load_mind2web_dataset(split=split_name)
        mind2web_total += len(mind2web_dataset.tasks)

        for task in mind2web_dataset.tasks:
            mind2web_domain_counts[task.domain] += 1
            num_steps = task.get_num_steps()
            task_length = len(task.confirmed_task.split())
            difficulty = num_steps * 2.0 + task_length * 0.1
            mind2web_difficulties.append({'difficulty': difficulty, 'domain': task.domain})

    mind2web_sorted = sorted(mind2web_domain_counts.items(), key=lambda x: x[1], reverse=True)
    mind2web_easy_thresh = np.percentile([d['difficulty'] for d in mind2web_difficulties], 33)
    mind2web_hard_thresh = np.percentile([d['difficulty'] for d in mind2web_difficulties], 67)

    return {
        'swebench': {
            'total': len(swebench_dataset.tasks),
            'sorted': swebench_sorted,
            'difficulties': swebench_difficulties,
            'thresholds': (swebench_easy_thresh, swebench_hard_thresh)
        },
        'webarena': {
            'total': len(webarena_tasks),
            'sorted': webarena_sorted,
            'difficulties': webarena_difficulties,
            'thresholds': (webarena_easy_thresh, webarena_hard_thresh)
        },
        'mind2web': {
            'total': mind2web_total,
            'sorted': mind2web_sorted,
            'difficulties': mind2web_difficulties,
            'thresholds': (mind2web_easy_thresh, mind2web_hard_thresh)
        }
    }

def create_combined_distributions(data, output_dir):
    """Create combined distribution plot for all benchmarks."""
    output_path = Path(output_dir)

    # Compact figure for 3 benchmarks
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, hspace=0.55)

    # SWE-bench
    ax1 = fig.add_subplot(gs[0])
    swebench_data = data['swebench']
    repos = [r.split('/')[-1] for r, _ in swebench_data['sorted']]
    swe_counts = [c for _, c in swebench_data['sorted']]
    swe_total = swebench_data['total']
    max_swe = max(swe_counts) if swe_counts else 0

    y_pos = np.arange(len(repos))
    bars = ax1.barh(
        y_pos,
        swe_counts,
        color='#6699CC',
        edgecolor='black',
        linewidth=1.0
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(repos, fontsize=8)
    ax1.set_xlabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'(a) SWE-bench: Repository Distribution ({swe_total} tasks, {len(repos)} repos)',
        fontsize=11,
        fontweight='bold',
        loc='left',
        pad=10,
    )
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_linewidth(1.0)
    ax1.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # label offset scales with data
    offset = max(2, max_swe * 0.01)
    for i, count in enumerate(swe_counts):
        ax1.text(
            count + offset,
            i,
            str(count),
            va='center',
            fontsize=7.5,
            fontweight='bold'
        )

    # WebArena
    ax2 = fig.add_subplot(gs[1])
    webarena_data = data['webarena']
    subsets = [s for s, _ in webarena_data['sorted']]
    wa_counts = [c for _, c in webarena_data['sorted']]
    wa_total = webarena_data['total']
    max_wa = max(wa_counts) if wa_counts else 0

    x_pos = np.arange(len(subsets))
    bars = ax2.bar(
        x_pos,
        wa_counts,
        color='#77AA77',
        width=0.6,
        edgecolor='black',
        linewidth=1.0
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(subsets, fontsize=9)
    ax2.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax2.set_title(
        f'(b) WebArena: Subset Distribution ({wa_total} tasks, {len(subsets)} subsets)',
        fontsize=11,
        fontweight='bold',
        loc='left',
        pad=15,
    )
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.0)
    ax2.spines['bottom'].set_linewidth(1.0)
    ax2.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    y_offset = max(3, max_wa * 0.012)
    for i, count in enumerate(wa_counts):
        pct = (count / wa_total) * 100 if wa_total > 0 else 0
        ax2.text(
            i,
            count + y_offset,
            f'{count}\n({pct:.0f}%)',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

    # Mind2Web - Combined (all three test splits)
    ax3 = fig.add_subplot(gs[2])
    mind2web_data = data['mind2web']
    domains = [d for d, _ in mind2web_data['sorted']]
    m2w_counts = [c for _, c in mind2web_data['sorted']]
    m2w_total = mind2web_data['total']
    max_m2w = max(m2w_counts) if m2w_counts else 0

    x_pos = np.arange(len(domains))
    bars = ax3.bar(
        x_pos,
        m2w_counts,
        color='#AA7799',
        width=0.5,
        edgecolor='black',
        linewidth=1.0
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(domains, fontsize=9)
    ax3.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
    ax3.set_title(
        f'(c) Mind2Web: Domain Distribution ({m2w_total} tasks, {len(domains)} domains)',
        fontsize=11,
        fontweight='bold',
        loc='left',
        pad=15,
    )
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1.0)
    ax3.spines['bottom'].set_linewidth(1.0)
    ax3.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
    ax3.set_axisbelow(True)

    y_offset = max(2, max_m2w * 0.02)
    for i, count in enumerate(m2w_counts):
        pct = (count / m2w_total) * 100 if m2w_total > 0 else 0
        ax3.text(
            i,
            count + y_offset,
            f'{count}\n({pct:.0f}%)',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold'
        )

    plt.tight_layout(pad=2.0)
    output_file = output_path / "combined_all_distributions.pdf"
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        format='pdf',
        pad_inches=0.25,
    )
    plt.savefig(
        output_path / "combined_all_distributions.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.25,
    )
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_combined_difficulty_overview(data, output_dir):
    """Create combined difficulty overview with distributions."""
    output_path = Path(output_dir)

    # Compact layout for 3 benchmarks
    fig = plt.figure(figsize=(11.5, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.2], wspace=0.35)

    datasets = [
        ('SWE-bench', data['swebench']),
        ('WebArena', data['webarena']),
        ('Mind2Web', data['mind2web']),
    ]

    # Compute shared max for consistent y-axis
    max_val = 0
    difficulty_counts = {}
    for name, d in datasets:
        easy = sum(1 for t in d['difficulties'] if t['difficulty'] <= d['thresholds'][0])
        medium = sum(1 for t in d['difficulties']
                     if d['thresholds'][0] < t['difficulty'] <= d['thresholds'][1])
        hard = sum(1 for t in d['difficulties'] if t['difficulty'] > d['thresholds'][1])
        difficulty_counts[name] = (easy, medium, hard)
        max_val = max(max_val, easy, medium, hard)

    y_max = max_val * 1.12 if max_val > 0 else 1
    label_offset = max_val * 0.035 if max_val > 0 else 0.8

    shared_ax = None
    for idx, (name, d) in enumerate(datasets):
        if idx == 0:
            ax = fig.add_subplot(gs[idx])
            shared_ax = ax
        else:
            ax = fig.add_subplot(gs[idx], sharey=shared_ax)

        easy, medium, hard = difficulty_counts[name]
        x = np.arange(3)
        values = [easy, medium, hard]
        colors = [COLORS['easy'], COLORS['medium'], COLORS['hard']]

        bars = ax.bar(
            x,
            values,
            color=colors,
            width=0.55,
            edgecolor='black',
            linewidth=1.0,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(['Easy', 'Medium', 'Hard'], fontsize=10)
        if idx == 0:
            ax.set_ylabel('Number of Tasks', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')

        ax.set_title(
            f'{name}\n({d["total"]} tasks)',
            fontsize=11,
            fontweight='bold',
            pad=8,
        )
        ax.set_ylim(0, y_max)

        for bar, val in zip(bars, values):
            pct = (val / d['total']) * 100 if d['total'] > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + label_offset,
                f'{val}\n({pct:.0f}%)',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # small panel label in the upper left of each axis
        panel_label = chr(ord('a') + idx)
        ax.text(
            -0.25,
            1.02,
            f'({panel_label})',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='bottom',
        )

    # Combined summary table (column 3)
    ax_table = fig.add_subplot(gs[3])
    ax_table.axis('off')

    table_data = [['Benchmark', 'Easy', 'Med', 'Hard', 'Total']]
    for name, d in datasets:
        easy, medium, hard = difficulty_counts[name]
        easy_pct = f"{easy}\n({easy / d['total'] * 100:.0f}%)" if d['total'] > 0 else f"{easy}\n(0%)"
        med_pct = f"{medium}\n({medium / d['total'] * 100:.0f}%)" if d['total'] > 0 else f"{medium}\n(0%)"
        hard_pct = f"{hard}\n({hard / d['total'] * 100:.0f}%)" if d['total'] > 0 else f"{hard}\n(0%)"
        table_data.append([name, easy_pct, med_pct, hard_pct, str(d['total'])])

    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.13, 1.35, 0.8],
        colWidths=[0.65, 0.30, 0.30, 0.30, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.2)

    # Style header row
    for j in range(5):
        cell = table[(0, j)]
        cell.set_facecolor('#DDDDDD')
        cell.set_text_props(weight='bold', fontsize=11)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

    # Style data rows with soft difficulty colors
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            cell.set_edgecolor('black')
            cell.set_linewidth(1.0)
            if j == 1:
                cell.set_facecolor('#E6F0FF')  # easy
            elif j == 2:
                cell.set_facecolor('#FFF4E6')  # medium
            elif j == 3:
                cell.set_facecolor('#FFE6E6')  # hard

    ax_table.text(
        0.5,
        1.0,
        'Summary Statistics',
        ha='center',
        va='top',
        fontsize=13,
        fontweight='bold',
        transform=ax_table.transAxes,
    )

    plt.suptitle(
        'Task Difficulty Distribution Across Benchmarks',
        fontsize=14,
        fontweight='bold',
        y=1.05,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.92], pad=2.0)

    output_file = output_path / "combined_difficulty_overview.pdf"
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        format='pdf',
        pad_inches=0.35,
    )
    plt.savefig(
        output_path / "combined_difficulty_overview.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.35,
    )
    logger.info(f"Saved: {output_file}")
    plt.close()



def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("CREATING COMBINED MULTI-BENCHMARK PLOTS")
    logger.info("="*70)

    # Create output directory
    output_dir = Path("plots/combined_plots_final")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Analyze all datasets
    data = analyze_all()

    # Create combined plots
    logger.info("\nGenerating combined plots...")
    create_combined_distributions(data, output_dir)
    create_combined_difficulty_overview(data, output_dir)

    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)
    logger.info(f"\nCombined plots saved to: {output_dir}/")
    logger.info("\nGenerated files:")
    logger.info("  1. combined_all_distributions.pdf/png - All benchmark distributions in one figure")
    logger.info("  2. combined_difficulty_overview.pdf/png - Difficulty comparison with summary table")


if __name__ == "__main__":
    main()
