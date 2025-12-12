#!/usr/bin/env python3
"""
Combine memory gap visualizations into a single publication-quality figure
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/hivamoh/Desktop/ReasoningBank")

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5

def create_combined_horizontal():
    """Create side-by-side layout (best for wide figures)"""

    # Load images
    img1 = mpimg.imread(str(BASE_DIR / "memory_gap_visualization.png"))
    img2 = mpimg.imread(str(BASE_DIR / "memory_funnel_visualization.png"))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Display images
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('(a) Retrieval-Utilization Gap',
                  fontsize=14, fontweight='bold', pad=10, loc='left')

    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('(b) Memory Utilization Funnel',
                  fontsize=14, fontweight='bold', pad=10, loc='left')

    # Add main title
    fig.suptitle('Memory System Performance: Retrieval vs. Utilization Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = BASE_DIR / "combined_memory_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Saved horizontal layout: {output_path}")

    output_path_png = BASE_DIR / "combined_memory_analysis.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Saved horizontal layout: {output_path_png}")

    return fig

def create_combined_vertical():
    """Create stacked layout (best for single-column papers)"""

    # Load images
    img1 = mpimg.imread(str(BASE_DIR / "memory_gap_visualization.png"))
    img2 = mpimg.imread(str(BASE_DIR / "memory_funnel_visualization.png"))

    # Create figure with vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))

    # Display images
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('(a) Retrieval-Utilization Gap: Bar Chart View',
                  fontsize=13, fontweight='bold', pad=10, loc='left')

    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('(b) Memory Utilization Funnel: Detailed Breakdown',
                  fontsize=13, fontweight='bold', pad=10, loc='left')

    # Add main title
    fig.suptitle('Memory System Performance Analysis\nFrom logs_pro: Gemini-2.5-Pro on WebArena Reddit',
                fontsize=14, fontweight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_path = BASE_DIR / "combined_memory_analysis_vertical.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Saved vertical layout: {output_path}")

    output_path_png = BASE_DIR / "combined_memory_analysis_vertical.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Saved vertical layout: {output_path_png}")

    return fig

def create_combined_compact():
    """Create compact layout with better spacing (RECOMMENDED)"""

    # Load images
    img1 = mpimg.imread(str(BASE_DIR / "memory_gap_visualization.png"))
    img2 = mpimg.imread(str(BASE_DIR / "memory_funnel_visualization.png"))

    # Create figure with custom gridspec
    fig = plt.figure(figsize=(14, 10))

    # Create grid: 2 rows with height ratio 1.1:1
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Display images
    ax1.imshow(img1)
    ax1.axis('off')

    # Add panel label with background
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
            fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=2))

    ax2.imshow(img2)
    ax2.axis('off')

    # Add panel label with background
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
            fontsize=16, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=2))

    # Add figure title (no suptitle, will be in caption)

    # Save
    output_path = BASE_DIR / "combined_memory_analysis_compact.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"✅ Saved compact layout: {output_path}")

    output_path_png = BASE_DIR / "combined_memory_analysis_compact.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"✅ Saved compact layout: {output_path_png}")

    return fig

def create_combined_side_by_side_clean():
    """Create clean side-by-side layout with minimal spacing (BEST FOR PAPER)"""

    # Load images
    img1 = mpimg.imread(str(BASE_DIR / "memory_gap_visualization.png"))
    img2 = mpimg.imread(str(BASE_DIR / "memory_funnel_visualization.png"))

    # Create figure
    fig = plt.figure(figsize=(16, 7))

    # Create grid: 1 row, 2 columns
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.02, right=0.98)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Display images
    ax1.imshow(img1)
    ax1.axis('off')

    # Add panel label (a)
    ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor='black', linewidth=2.5, alpha=0.95))

    ax2.imshow(img2)
    ax2.axis('off')

    # Add panel label (b)
    ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                     edgecolor='black', linewidth=2.5, alpha=0.95))

    # Save with high quality
    output_path = BASE_DIR / "combined_memory_analysis_final.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.05)
    print(f"✅ Saved final side-by-side: {output_path}")

    output_path_png = BASE_DIR / "combined_memory_analysis_final.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.05)
    print(f"✅ Saved final side-by-side: {output_path_png}")

    return fig

def create_publication_ready():
    """Create the ultimate publication-ready version with perfect spacing"""

    # Load images
    img1 = mpimg.imread(str(BASE_DIR / "memory_gap_visualization.png"))
    img2 = mpimg.imread(str(BASE_DIR / "memory_funnel_visualization.png"))

    # Create figure - optimized for two-column paper
    fig = plt.figure(figsize=(14, 8.5))

    # Create grid
    gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.03, right=0.97,
                         top=0.95, bottom=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Display images
    ax1.imshow(img1)
    ax1.axis('off')

    # Panel (a) label - professional style
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left',
            color='black',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                     edgecolor='black', linewidth=3, alpha=0.98))

    ax2.imshow(img2)
    ax2.axis('off')

    # Panel (b) label - professional style
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left',
            color='black',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                     edgecolor='black', linewidth=3, alpha=0.98))

    # Save with maximum quality
    output_path = BASE_DIR / "Figure_MemoryGapAnalysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none',
               pad_inches=0.08, transparent=False)
    print(f"\n⭐ PUBLICATION-READY VERSION: {output_path}")

    output_path_png = BASE_DIR / "Figure_MemoryGapAnalysis.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none',
               pad_inches=0.08, transparent=False)
    print(f"⭐ PUBLICATION-READY VERSION: {output_path_png}")

    plt.close()

    return output_path

def main():
    """Generate all combined layouts"""
    print("="*70)
    print("COMBINING MEMORY VISUALIZATIONS")
    print("="*70)

    print("\nGenerating multiple layout options...")

    # Generate all layouts
    create_combined_horizontal()
    print()
    create_combined_vertical()
    print()
    create_combined_compact()
    print()
    create_combined_side_by_side_clean()
    print()
    final_path = create_publication_ready()

    print("\n" + "="*70)
    print("ALL LAYOUTS GENERATED")
    print("="*70)

    print("\n📊 Available layouts:")
    print("  1. combined_memory_analysis.pdf - Side-by-side with title")
    print("  2. combined_memory_analysis_vertical.pdf - Stacked vertically")
    print("  3. combined_memory_analysis_compact.pdf - Compact stacked")
    print("  4. combined_memory_analysis_final.pdf - Clean side-by-side")
    print("  5. Figure_MemoryGapAnalysis.pdf ⭐ RECOMMENDED for paper")

    print("\n🎯 RECOMMENDED FOR YOUR PAPER:")
    print(f"  Use: Figure_MemoryGapAnalysis.pdf")
    print(f"  Location: {final_path}")

    print("\n📝 LaTeX code:")
    print("""
\\begin{figure*}[t]
\\centering
\\includegraphics[width=0.95\\textwidth]{Figure_MemoryGapAnalysis.pdf}
\\caption{Memory system performance analysis from logs\\_pro (Gemini-2.5-Pro).
(a) Retrieval-utilization gap showing that while 98.6\\% of retrieved memories
are relevant (high precision), only 15\\% measurably influence agent reasoning.
The visualization depicts the flow from query through k=3 retrieval to utilization,
with green circles representing retrieved memories, red X marks indicating unused
memories, and a dashed red box highlighting the critical 83.6 percentage-point gap.
(b) Detailed funnel breakdown showing progressive memory drop-off at each stage:
from 100\\% queries to 98.6\\% relevant, 45\\% contextually appropriate, 15\\%
actually utilized in reasoning, and only 8\\% influencing final actions.
The largest loss (53\\%) occurs between contextual appropriateness and actual
utilization, revealing the key bottleneck in memory integration.}
\\label{fig:memory-gap-analysis}
\\end{figure*}

% Note: Use figure* for two-column papers to span both columns
% For single-column, use figure instead of figure*
    """)

    print("\n✅ All files saved in:", BASE_DIR)

if __name__ == "__main__":
    main()
