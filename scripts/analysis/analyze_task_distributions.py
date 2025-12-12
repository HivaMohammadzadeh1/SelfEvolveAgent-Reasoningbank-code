#!/usr/bin/env python3
"""
Task Distribution Analysis for SWE-bench, WebArena, and Mind2Web

Publication-quality visualizations with difficulty analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from pathlib import Path
from loguru import logger
import numpy as np
import sys

# Import dataset loaders
from src.swebench_loader import load_swebench_dataset
from src.webarena_loader import WebArenaDataset
from src.mind2web_loader import load_mind2web_dataset

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def analyze_swebench():
    """Analyze SWE-bench task distribution and difficulty."""
    logger.info("\n" + "="*70)
    logger.info("ANALYZING SWE-BENCH TASK DISTRIBUTION")
    logger.info("="*70)

    try:
        dataset = load_swebench_dataset()

        # Count tasks by repository
        repo_counts = Counter()
        task_difficulties = []

        for task in dataset.tasks:
            repo_counts[task.repo] += 1

            # Estimate difficulty based on patch size and problem statement length
            patch_lines = len(task.gold_patch.split('\n')) if task.gold_patch else 0
            problem_length = len(task.problem_statement.split())

            # Difficulty score: combination of patch complexity and problem length
            difficulty_score = patch_lines * 0.5 + problem_length * 0.1

            task_difficulties.append({
                'repo': task.repo,
                'patch_lines': patch_lines,
                'problem_words': problem_length,
                'difficulty': difficulty_score,
                'instance_id': task.instance_id
            })

        sorted_repos = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)

        # Difficulty thresholds (using percentiles)
        difficulties = [t['difficulty'] for t in task_difficulties]
        easy_threshold = np.percentile(difficulties, 33)
        hard_threshold = np.percentile(difficulties, 67)

        easy_tasks = [t for t in task_difficulties if t['difficulty'] <= easy_threshold]
        medium_tasks = [t for t in task_difficulties if easy_threshold < t['difficulty'] <= hard_threshold]
        hard_tasks = [t for t in task_difficulties if t['difficulty'] > hard_threshold]

        logger.info(f"\nTotal tasks: {len(dataset.tasks)}")
        logger.info(f"Total repositories: {len(repo_counts)}")
        logger.info(f"\nDifficulty Distribution:")
        logger.info(f"  Easy tasks:   {len(easy_tasks)} ({len(easy_tasks)/len(dataset.tasks)*100:.1f}%)")
        logger.info(f"  Medium tasks: {len(medium_tasks)} ({len(medium_tasks)/len(dataset.tasks)*100:.1f}%)")
        logger.info(f"  Hard tasks:   {len(hard_tasks)} ({len(hard_tasks)/len(dataset.tasks)*100:.1f}%)")

        logger.info(f"\nDifficulty Metrics:")
        logger.info(f"  Mean patch lines: {np.mean([t['patch_lines'] for t in task_difficulties]):.1f}")
        logger.info(f"  Mean problem length: {np.mean([t['problem_words'] for t in task_difficulties]):.1f} words")

        return {
            'total_tasks': len(dataset.tasks),
            'total_repos': len(repo_counts),
            'repo_counts': dict(sorted_repos),
            'sorted_repos': sorted_repos,
            'difficulties': task_difficulties,
            'easy_tasks': easy_tasks,
            'medium_tasks': medium_tasks,
            'hard_tasks': hard_tasks,
            'difficulty_thresholds': (easy_threshold, hard_threshold)
        }

    except Exception as e:
        logger.error(f"Failed to analyze SWE-bench: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_webarena():
    """Analyze WebArena task distribution and difficulty."""
    logger.info("\n" + "="*70)
    logger.info("ANALYZING WEBARENA TASK DISTRIBUTION")
    logger.info("="*70)

    try:
        dataset = WebArenaDataset(data_dir="data/webarena")
        # Exclude 'map' subset, limit multi to 29 - standard 684 tasks as per PRD
        tasks = dataset.load_tasks(
            subsets=["shopping", "admin", "gitlab", "reddit", "multi"],
            max_multi_tasks=29
        )

        subset_counts = Counter()
        task_difficulties = []

        for task in tasks:
            subset_counts[task['subset']] += 1

            # More nuanced difficulty scoring for WebArena (matching combined plots)
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

            task_difficulties.append({
                'task_id': task['task_id'],
                'subset': task['subset'],
                'difficulty': difficulty,
                'require_login': task.get('require_login', False),
            })

        sorted_subsets = sorted(subset_counts.items(), key=lambda x: x[1], reverse=True)

        # Difficulty analysis
        difficulties = [t['difficulty'] for t in task_difficulties]
        easy_threshold = np.percentile(difficulties, 33)
        hard_threshold = np.percentile(difficulties, 67)

        easy_tasks = [t for t in task_difficulties if t['difficulty'] <= easy_threshold]
        medium_tasks = [t for t in task_difficulties if easy_threshold < t['difficulty'] <= hard_threshold]
        hard_tasks = [t for t in task_difficulties if t['difficulty'] > hard_threshold]

        logger.info(f"\nTotal tasks: {len(tasks)}")
        logger.info(f"Total subsets: {len(subset_counts)}")

        logger.info(f"\nDifficulty Distribution:")
        logger.info(f"  Easy tasks:   {len(easy_tasks)} ({len(easy_tasks)/len(tasks)*100:.1f}%)")
        logger.info(f"  Medium tasks: {len(medium_tasks)} ({len(medium_tasks)/len(tasks)*100:.1f}%)")
        logger.info(f"  Hard tasks:   {len(hard_tasks)} ({len(hard_tasks)/len(tasks)*100:.1f}%)")

        logger.info(f"\nTask Distribution by Subset:")
        for subset, count in sorted_subsets:
            pct = (count / len(tasks)) * 100
            subset_hard = len([t for t in hard_tasks if t['subset'] == subset])
            logger.info(f"  {subset:10s}: {count:3d} tasks ({pct:5.1f}%) - {subset_hard} hard tasks")

        return {
            'total_tasks': len(tasks),
            'total_subsets': len(subset_counts),
            'subset_counts': dict(sorted_subsets),
            'sorted_subsets': sorted_subsets,
            'difficulties': task_difficulties,
            'easy_tasks': easy_tasks,
            'medium_tasks': medium_tasks,
            'hard_tasks': hard_tasks,
            'difficulty_thresholds': (easy_threshold, hard_threshold)
        }

    except Exception as e:
        logger.error(f"Failed to analyze WebArena: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_mind2web(split="test_task"):
    """Analyze Mind2Web task distribution and difficulty - Load all three test splits combined."""
    logger.info("\n" + "="*70)
    logger.info(f"ANALYZING MIND2WEB TASK DISTRIBUTION (Combined: test_task + test_website + test_domain)")
    logger.info("="*70)

    try:
        # Load all three test splits and combine them
        domain_counts = Counter()
        website_counts = Counter()
        subdomain_counts = Counter()
        task_difficulties = []
        total_tasks = 0

        for split_name in ["test_task", "test_website", "test_domain"]:
            dataset = load_mind2web_dataset(split=split_name)
            total_tasks += len(dataset.tasks)

            for task in dataset.tasks:
                domain_counts[task.domain] += 1
                website_counts[task.website] += 1
                subdomain_counts[task.subdomain] += 1

                # Difficulty based on number of action steps (ground truth)
                num_steps = task.get_num_steps()

                # Additional factors
                task_length = len(task.confirmed_task.split())

                difficulty_score = num_steps * 2.0 + task_length * 0.1

                task_difficulties.append({
                    'annotation_id': task.annotation_id,
                    'website': task.website,
                    'domain': task.domain,
                    'subdomain': task.subdomain,
                    'num_steps': num_steps,
                    'task_length': task_length,
                    'difficulty': difficulty_score
                })

        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_websites = sorted(website_counts.items(), key=lambda x: x[1], reverse=True)

        # Difficulty analysis
        difficulties = [t['difficulty'] for t in task_difficulties]
        easy_threshold = np.percentile(difficulties, 33)
        hard_threshold = np.percentile(difficulties, 67)

        easy_tasks = [t for t in task_difficulties if t['difficulty'] <= easy_threshold]
        medium_tasks = [t for t in task_difficulties if easy_threshold < t['difficulty'] <= hard_threshold]
        hard_tasks = [t for t in task_difficulties if t['difficulty'] > hard_threshold]

        logger.info(f"\nTotal tasks: {total_tasks}")
        logger.info(f"Total domains: {len(domain_counts)}")
        logger.info(f"Total websites: {len(website_counts)}")

        logger.info(f"\nDifficulty Distribution:")
        logger.info(f"  Easy tasks:   {len(easy_tasks)} ({len(easy_tasks)/total_tasks*100:.1f}%)")
        logger.info(f"  Medium tasks: {len(medium_tasks)} ({len(medium_tasks)/total_tasks*100:.1f}%)")
        logger.info(f"  Hard tasks:   {len(hard_tasks)} ({len(hard_tasks)/total_tasks*100:.1f}%)")

        logger.info(f"\nDifficulty Metrics:")
        logger.info(f"  Mean steps: {np.mean([t['num_steps'] for t in task_difficulties]):.1f}")
        logger.info(f"  Mean task length: {np.mean([t['task_length'] for t in task_difficulties]):.1f} words")

        return {
            'total_tasks': total_tasks,
            'total_domains': len(domain_counts),
            'total_websites': len(website_counts),
            'domain_counts': dict(sorted_domains),
            'website_counts': dict(sorted_websites),
            'sorted_domains': sorted_domains,
            'sorted_websites': sorted_websites,
            'difficulties': task_difficulties,
            'easy_tasks': easy_tasks,
            'medium_tasks': medium_tasks,
            'hard_tasks': hard_tasks,
            'difficulty_thresholds': (easy_threshold, hard_threshold)
        }

    except Exception as e:
        logger.error(f"Failed to analyze Mind2Web: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_swebench_publication(data, output_dir="plots"):
    """Create publication-quality SWE-bench visualizations."""
    if not data:
        return

    output_path = Path(output_dir)

    # Figure 1: Repository distribution with difficulty
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Repository distribution
    top_n = 12  # All repos
    repos = [r.split('/')[-1] for r, _ in data['sorted_repos'][:top_n]]
    counts = [c for _, c in data['sorted_repos'][:top_n]]

    # Count difficulty by repo
    repo_difficulties = {r: {'easy': 0, 'medium': 0, 'hard': 0} for r, _ in data['sorted_repos'][:top_n]}
    for task in data['difficulties']:
        if task['repo'] in [r for r, _ in data['sorted_repos'][:top_n]]:
            if task['difficulty'] <= data['difficulty_thresholds'][0]:
                repo_difficulties[task['repo']]['easy'] += 1
            elif task['difficulty'] <= data['difficulty_thresholds'][1]:
                repo_difficulties[task['repo']]['medium'] += 1
            else:
                repo_difficulties[task['repo']]['hard'] += 1

    # Stacked bar chart
    easy_counts = [repo_difficulties[r]['easy'] for r, _ in data['sorted_repos'][:top_n]]
    medium_counts = [repo_difficulties[r]['medium'] for r, _ in data['sorted_repos'][:top_n]]
    hard_counts = [repo_difficulties[r]['hard'] for r, _ in data['sorted_repos'][:top_n]]

    y_pos = np.arange(len(repos))

    ax1.barh(y_pos, easy_counts, color='#2ecc71', label='Easy', alpha=0.9)
    ax1.barh(y_pos, medium_counts, left=easy_counts, color='#f39c12', label='Medium', alpha=0.9)
    ax1.barh(y_pos, hard_counts, left=np.array(easy_counts)+np.array(medium_counts),
             color='#e74c3c', label='Hard', alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(repos, fontsize=14)
    ax1.set_xlabel('Number of Tasks', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Task Distribution by Repository', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=14, frameon=True, shadow=True)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add total counts
    for i, (count, repo) in enumerate(zip(counts, repos)):
        ax1.text(count + 3, i, str(count), va='center', fontsize=13, fontweight='bold')

    # Difficulty distribution pie chart
    sizes = [len(data['easy_tasks']), len(data['medium_tasks']), len(data['hard_tasks'])]
    labels = [f"Easy\n({sizes[0]} tasks)", f"Medium\n({sizes[1]} tasks)", f"Hard\n({sizes[2]} tasks)"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=explode, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                        pctdistance=0.85)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(15)
        autotext.set_fontweight('bold')

    ax2.set_title('(b) Difficulty Distribution', fontsize=18, fontweight='bold', pad=15)

    plt.suptitle(f'SWE-bench: {data["total_tasks"]} Tasks across {data["total_repos"]} Repositories',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = output_path / "swebench_publication.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_webarena_publication(data, output_dir="plots"):
    """Create publication-quality WebArena visualizations."""
    if not data:
        return

    output_path = Path(output_dir)

    # Figure: Subset distribution with difficulty
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Subset distribution with difficulty breakdown
    subsets = [s for s, _ in data['sorted_subsets']]
    counts = [c for _, c in data['sorted_subsets']]

    # Count difficulty by subset
    subset_difficulties = {s: {'easy': 0, 'medium': 0, 'hard': 0} for s in subsets}
    for task in data['difficulties']:
        if task['subset'] in subsets:
            if task['difficulty'] <= data['difficulty_thresholds'][0]:
                subset_difficulties[task['subset']]['easy'] += 1
            elif task['difficulty'] <= data['difficulty_thresholds'][1]:
                subset_difficulties[task['subset']]['medium'] += 1
            else:
                subset_difficulties[task['subset']]['hard'] += 1

    # Stacked bar chart
    easy_counts = [subset_difficulties[s]['easy'] for s in subsets]
    medium_counts = [subset_difficulties[s]['medium'] for s in subsets]
    hard_counts = [subset_difficulties[s]['hard'] for s in subsets]

    x_pos = np.arange(len(subsets))
    width = 0.7

    ax1.bar(x_pos, easy_counts, width, color='#2ecc71', label='Easy', alpha=0.9)
    ax1.bar(x_pos, medium_counts, width, bottom=easy_counts, color='#f39c12', label='Medium', alpha=0.9)
    ax1.bar(x_pos, hard_counts, width, bottom=np.array(easy_counts)+np.array(medium_counts),
            color='#e74c3c', label='Hard', alpha=0.9)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(subsets, fontsize=15, fontweight='bold')
    ax1.set_ylabel('Number of Tasks', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Task Distribution by Subset', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=14, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels and percentages
    for i, (subset, count) in enumerate(zip(subsets, counts)):
        pct = (count / data['total_tasks']) * 100
        ax1.text(i, count + 5, f"{count}\n({pct:.1f}%)", ha='center', va='bottom',
                fontsize=13, fontweight='bold')

    # Difficulty distribution pie chart
    sizes = [len(data['easy_tasks']), len(data['medium_tasks']), len(data['hard_tasks'])]
    labels = [f"Easy\n({sizes[0]} tasks)", f"Medium\n({sizes[1]} tasks)", f"Hard\n({sizes[2]} tasks)"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=explode, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                        pctdistance=0.85)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(15)
        autotext.set_fontweight('bold')

    ax2.set_title('(b) Difficulty Distribution', fontsize=18, fontweight='bold', pad=15)

    plt.suptitle(f'WebArena: {data["total_tasks"]} Tasks across {data["total_subsets"]} Subsets',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = output_path / "webarena_publication.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_mind2web_publication(data, output_dir="plots"):
    """Create publication-quality Mind2Web visualizations."""
    if not data:
        return

    output_path = Path(output_dir)

    # Figure 1: Domain distribution
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Domain distribution with difficulty
    domains = [d for d, _ in data['sorted_domains']]
    counts = [c for _, c in data['sorted_domains']]

    # Count difficulty by domain
    domain_difficulties = {d: {'easy': 0, 'medium': 0, 'hard': 0} for d in domains}
    for task in data['difficulties']:
        if task['domain'] in domains:
            if task['difficulty'] <= data['difficulty_thresholds'][0]:
                domain_difficulties[task['domain']]['easy'] += 1
            elif task['difficulty'] <= data['difficulty_thresholds'][1]:
                domain_difficulties[task['domain']]['medium'] += 1
            else:
                domain_difficulties[task['domain']]['hard'] += 1

    # Stacked bar chart
    easy_counts = [domain_difficulties[d]['easy'] for d in domains]
    medium_counts = [domain_difficulties[d]['medium'] for d in domains]
    hard_counts = [domain_difficulties[d]['hard'] for d in domains]

    x_pos = np.arange(len(domains))
    width = 0.6

    ax1.bar(x_pos, easy_counts, width, color='#2ecc71', label='Easy', alpha=0.9)
    ax1.bar(x_pos, medium_counts, width, bottom=easy_counts, color='#f39c12', label='Medium', alpha=0.9)
    ax1.bar(x_pos, hard_counts, width, bottom=np.array(easy_counts)+np.array(medium_counts),
            color='#e74c3c', label='Hard', alpha=0.9)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(domains, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Tasks', fontsize=16, fontweight='bold')
    ax1.set_title('(a) Task Distribution by Domain', fontsize=18, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=14, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels and percentages
    for i, (domain, count) in enumerate(zip(domains, counts)):
        pct = (count / data['total_tasks']) * 100
        ax1.text(i, count + 2, f"{count}\n({pct:.1f}%)", ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    # Difficulty distribution pie chart
    sizes = [len(data['easy_tasks']), len(data['medium_tasks']), len(data['hard_tasks'])]
    labels = [f"Easy\n({sizes[0]} tasks)", f"Medium\n({sizes[1]} tasks)", f"Hard\n({sizes[2]} tasks)"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=explode, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                        pctdistance=0.85)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(15)
        autotext.set_fontweight('bold')

    ax2.set_title('(b) Difficulty Distribution', fontsize=18, fontweight='bold', pad=15)

    plt.suptitle(f'Mind2Web: {data["total_tasks"]} Tasks across {data["total_domains"]} Domains '
                 f'({data["total_websites"]} Websites)',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = output_path / "mind2web_publication.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_combined_overview(swebench_data, webarena_data, mind2web_data, output_dir="plots"):
    """Create publication-quality combined overview."""
    output_path = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    datasets = [
        ("SWE-bench", swebench_data, '#3498db'),
        ("WebArena", webarena_data, '#27ae60'),
        ("Mind2Web", mind2web_data, '#9b59b6')
    ]

    # Find max value across all datasets for consistent y-axis
    max_value = 0
    all_values = []
    for name, data, color in datasets:
        if data:
            easy = len(data['easy_tasks'])
            medium = len(data['medium_tasks'])
            hard = len(data['hard_tasks'])
            all_values.extend([easy, medium, hard])
            max_value = max(max_value, easy, medium, hard)

    # Set consistent y-axis limit
    y_max = max_value * 1.25

    for idx, (name, data, color) in enumerate(datasets):
        if not data:
            continue

        ax = axes[idx]

        # Difficulty breakdown
        easy = len(data['easy_tasks'])
        medium = len(data['medium_tasks'])
        hard = len(data['hard_tasks'])

        categories = ['Easy', 'Medium', 'Hard']
        values = [easy, medium, hard]
        colors_diff = ['#2ecc71', '#f39c12', '#e74c3c']

        bars = ax.bar(categories, values, color=colors_diff, alpha=0.9, edgecolor='black', linewidth=2)

        ax.set_ylabel('Number of Tasks', fontsize=16, fontweight='bold')
        ax.set_title(f'{name}\n({data["total_tasks"]} total tasks)',
                    fontsize=18, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels and percentages
        for bar, value in zip(bars, values):
            height = bar.get_height()
            pct = (value / data['total_tasks']) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + max_value*0.02,
                   f'{value}\n({pct:.1f}%)', ha='center', va='bottom',
                   fontsize=14, fontweight='bold')

        # Set consistent y-axis limit across all subplots
        ax.set_ylim(0, y_max)

    plt.suptitle('Task Difficulty Distribution Across Benchmarks',
                fontsize=22, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = output_path / "combined_overview_publication.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_swebench_detailed(data, output_dir="plots"):
    """Create detailed SWE-bench repository distribution plot."""
    if not data:
        return

    output_path = Path(output_dir)

    fig = plt.figure(figsize=(18, 10))

    # All repositories
    repos = [r.split('/')[-1] for r, _ in data['sorted_repos']]
    counts = [c for _, c in data['sorted_repos']]

    # Count difficulty by repo
    repo_difficulties = {r: {'easy': 0, 'medium': 0, 'hard': 0} for r, _ in data['sorted_repos']}
    for task in data['difficulties']:
        if task['repo'] in [r for r, _ in data['sorted_repos']]:
            if task['difficulty'] <= data['difficulty_thresholds'][0]:
                repo_difficulties[task['repo']]['easy'] += 1
            elif task['difficulty'] <= data['difficulty_thresholds'][1]:
                repo_difficulties[task['repo']]['medium'] += 1
            else:
                repo_difficulties[task['repo']]['hard'] += 1

    # Stacked horizontal bar chart
    easy_counts = [repo_difficulties[r]['easy'] for r, _ in data['sorted_repos']]
    medium_counts = [repo_difficulties[r]['medium'] for r, _ in data['sorted_repos']]
    hard_counts = [repo_difficulties[r]['hard'] for r, _ in data['sorted_repos']]

    y_pos = np.arange(len(repos))

    plt.barh(y_pos, easy_counts, color='#2ecc71', label='Easy', alpha=0.9, edgecolor='black', linewidth=1)
    plt.barh(y_pos, medium_counts, left=easy_counts, color='#f39c12', label='Medium', alpha=0.9, edgecolor='black', linewidth=1)
    plt.barh(y_pos, hard_counts, left=np.array(easy_counts)+np.array(medium_counts),
             color='#e74c3c', label='Hard', alpha=0.9, edgecolor='black', linewidth=1)

    plt.yticks(y_pos, repos, fontsize=15)
    plt.xlabel('Number of Tasks', fontsize=18, fontweight='bold')
    plt.title(f'SWE-bench: Task Distribution by Repository\n{data["total_tasks"]} Tasks across {data["total_repos"]} Repositories',
             fontsize=20, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=16, frameon=True, shadow=True, fancybox=True)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')

    # Add total counts
    for i, count in enumerate(counts):
        plt.text(count + 3, i, f'{count}', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout(pad=1.5)
    output_file = output_path / "swebench_detailed_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_mind2web_detailed(data, output_dir="plots"):
    """Create detailed Mind2Web website distribution plot."""
    if not data:
        return

    output_path = Path(output_dir)

    fig = plt.figure(figsize=(18, 12))

    # Top 25 websites
    top_n = 25
    websites = [w for w, _ in data['sorted_websites'][:top_n]]
    counts = [c for _, c in data['sorted_websites'][:top_n]]

    # Count difficulty by website
    website_difficulties = {w: {'easy': 0, 'medium': 0, 'hard': 0} for w in websites}
    for task in data['difficulties']:
        if task['website'] in websites:
            if task['difficulty'] <= data['difficulty_thresholds'][0]:
                website_difficulties[task['website']]['easy'] += 1
            elif task['difficulty'] <= data['difficulty_thresholds'][1]:
                website_difficulties[task['website']]['medium'] += 1
            else:
                website_difficulties[task['website']]['hard'] += 1

    # Stacked horizontal bar chart
    easy_counts = [website_difficulties[w]['easy'] for w in websites]
    medium_counts = [website_difficulties[w]['medium'] for w in websites]
    hard_counts = [website_difficulties[w]['hard'] for w in websites]

    y_pos = np.arange(len(websites))

    plt.barh(y_pos, easy_counts, color='#2ecc71', label='Easy', alpha=0.9, edgecolor='black', linewidth=1)
    plt.barh(y_pos, medium_counts, left=easy_counts, color='#f39c12', label='Medium', alpha=0.9, edgecolor='black', linewidth=1)
    plt.barh(y_pos, hard_counts, left=np.array(easy_counts)+np.array(medium_counts),
             color='#e74c3c', label='Hard', alpha=0.9, edgecolor='black', linewidth=1)

    plt.yticks(y_pos, websites, fontsize=13)
    plt.xlabel('Number of Tasks', fontsize=18, fontweight='bold')
    plt.title(f'Mind2Web: Top {top_n} Websites by Task Count\n{data["total_tasks"]} Tasks from {data["total_websites"]} Websites',
             fontsize=20, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=16, frameon=True, shadow=True, fancybox=True)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')

    # Add total counts
    for i, count in enumerate(counts):
        plt.text(count + 0.2, i, f'{count}', va='center', fontsize=13, fontweight='bold')

    plt.tight_layout(pad=1.5)
    output_file = output_path / "mind2web_detailed_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function to run all analyses."""
    logger.info("="*70)
    logger.info("PUBLICATION-QUALITY TASK DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Analyze each dataset
    swebench_data = analyze_swebench()
    webarena_data = analyze_webarena()
    mind2web_data = analyze_mind2web(split="test_task")

    # Create visualizations
    logger.info("\n" + "="*70)
    logger.info("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
    logger.info("="*70)

    plot_swebench_publication(swebench_data, output_dir)
    plot_webarena_publication(webarena_data, output_dir)
    plot_mind2web_publication(mind2web_data, output_dir)
    plot_combined_overview(swebench_data, webarena_data, mind2web_data, output_dir)

    # Create detailed distribution plots
    plot_swebench_detailed(swebench_data, output_dir)
    plot_mind2web_detailed(mind2web_data, output_dir)

    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    if swebench_data:
        logger.info(f"\nSWE-bench ({swebench_data['total_tasks']} tasks):")
        logger.info(f"  - {swebench_data['total_repos']} repositories")
        logger.info(f"  - Easy: {len(swebench_data['easy_tasks'])} | "
                   f"Medium: {len(swebench_data['medium_tasks'])} | "
                   f"Hard: {len(swebench_data['hard_tasks'])}")

    if webarena_data:
        logger.info(f"\nWebArena ({webarena_data['total_tasks']} tasks):")
        logger.info(f"  - {webarena_data['total_subsets']} subsets")
        logger.info(f"  - Easy: {len(webarena_data['easy_tasks'])} | "
                   f"Medium: {len(webarena_data['medium_tasks'])} | "
                   f"Hard: {len(webarena_data['hard_tasks'])}")

    if mind2web_data:
        logger.info(f"\nMind2Web ({mind2web_data['total_tasks']} tasks):")
        logger.info(f"  - {mind2web_data['total_websites']} websites in {mind2web_data['total_domains']} domains")
        logger.info(f"  - Easy: {len(mind2web_data['easy_tasks'])} | "
                   f"Medium: {len(mind2web_data['medium_tasks'])} | "
                   f"Hard: {len(mind2web_data['hard_tasks'])}")

    logger.info(f"\nAll publication-quality plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
