"""
Comprehensive Exploratory Data Analysis (EDA) visualizations.
Generates preprocessing preview, class balance, subset distribution, and detailed metadata analysis.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import config, utils, preprocess


def viz_preprocessing():
    """Show preprocessing pipeline visualization."""
    print("\n" + "="*60)
    print("GENERATING PREPROCESSING PREVIEW")
    print("="*60)
    
    # Load metadata
    df = utils.load_metadata()
    
    # Generate preview (pass number of samples)
    preprocess.preview_preprocessing(n=6)
    print(f"✅ Preprocessing preview saved to {config.REPORTS_DIR}/preprocessing_preview.png")


def viz_class_balance():
    """Plot class distribution with detailed labels."""
    print("\n" + "="*60)
    print("GENERATING CLASS BALANCE CHART")
    print("="*60)
    
    # Load metadata
    df = utils.load_metadata()
    
    # Count classes
    class_counts = df[config.TARGET_COL].value_counts().sort_index()
    
    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_counts.index, class_counts.values, color=["steelblue", "coral"], alpha=0.8, edgecolor='black')
    
    # Add numeric labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel(config.TARGET_COL.capitalize(), fontsize=13, fontweight='bold')
    plt.ylabel("Count", fontsize=13, fontweight='bold')
    plt.title(f"Class Distribution: {config.TARGET_COL.capitalize()}", fontsize=15, fontweight='bold')
    plt.xticks([0, 1], ['Real (0)', 'Altered (1)'], fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/class_balance.png")
    print(f"✅ Class balance chart saved to {config.REPORTS_DIR}/class_balance.png")
    
    # Print statistics
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        cls_name = "Real" if cls == 0 else "Altered"
        print(f"  {cls_name} ({cls}): {count:,} ({count/len(df)*100:.2f}%)")


def viz_subset_distribution():
    """Plot subset distribution with detailed labels."""
    print("\n" + "="*60)
    print("GENERATING SUBSET DISTRIBUTION CHART")
    print("="*60)
    
    # Load metadata
    df = utils.load_metadata()
    
    # Count subsets (sort by order: Real, Altered-Easy, Medium, Hard)
    subset_order = ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
    subset_counts = df["subset"].value_counts()
    subset_counts = subset_counts.reindex([s for s in subset_order if s in subset_counts.index])
    
    # Plot
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'lightgreen', 'orange', 'crimson']
    bars = plt.bar(range(len(subset_counts)), subset_counts.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add numeric labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel("Subset", fontsize=13, fontweight='bold')
    plt.ylabel("Count", fontsize=13, fontweight='bold')
    plt.title("Dataset Subset Distribution", fontsize=15, fontweight='bold')
    plt.xticks(range(len(subset_counts)), subset_counts.index, rotation=30, ha='right', fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    utils.save_fig(f"{config.REPORTS_DIR}/subset_distribution.png")
    print(f"✅ Subset distribution chart saved to {config.REPORTS_DIR}/subset_distribution.png")
    
    # Print statistics
    print("\nSubset Distribution:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count:,} ({count/len(df)*100:.2f}%)")


def viz_comprehensive_eda():
    """Generate comprehensive EDA with multiple panels."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE EDA DASHBOARD")
    print("="*60)
    
    # Load metadata
    df = utils.load_metadata()
    
    # Create figure with subplots - increased spacing and height
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    
    # 1. Gender Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    gender_counts = df['gender'].value_counts()
    bars = ax1.bar(gender_counts.index, gender_counts.values, color=['skyblue', 'pink'], alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xlabel("Gender", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax1.set_title("Gender Distribution", fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylim(0, gender_counts.max() * 1.15)  # Add space for labels
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Hand Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    hand_counts = df['hand'].value_counts()
    bars = ax2.bar(hand_counts.index, hand_counts.values, color=['lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xlabel("Hand", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax2.set_title("Hand Distribution (Left/Right)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylim(0, hand_counts.max() * 1.15)  # Add space for labels
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Subject Count Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    subject_counts = df.groupby('subject_id').size()
    ax3.hist(subject_counts, bins=30, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax3.axvline(subject_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {subject_counts.mean():.1f}')
    ax3.axvline(subject_counts.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {subject_counts.median():.1f}')
    ax3.set_xlabel("Samples per Subject", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax3.set_title("Samples per Subject Distribution", fontsize=11, fontweight='bold', pad=10)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(alpha=0.3)
    
    # 4. Alteration Type Distribution (from filename patterns)
    ax4 = fig.add_subplot(gs[1, 0])
    # Extract alteration types from paths (CR, Obl, Zcut for altered images)
    altered_df = df[df['is_altered'] == 1].copy()
    alteration_types = altered_df['path'].str.extract(r'_(CR|Obl|Zcut)\.BMP')[0]
    alteration_counts = alteration_types.value_counts()
    if len(alteration_counts) > 0:
        bars = ax4.bar(alteration_counts.index, alteration_counts.values, 
                      color=['gold', 'lightblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    f'{int(height):,}\n({height/len(altered_df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_xlabel("Alteration Type", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax4.set_title("Alteration Types\n(Altered Images Only)", fontsize=11, fontweight='bold', pad=10)
    ax4.set_ylim(0, alteration_counts.max() * 1.15)  # Add space for labels
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Gender × Subset Cross-tabulation
    ax5 = fig.add_subplot(gs[1, 1])
    crosstab = pd.crosstab(df['subset'], df['gender'])
    subset_order = ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
    crosstab = crosstab.reindex([s for s in subset_order if s in crosstab.index])
    crosstab.plot(kind='bar', ax=ax5, color=['skyblue', 'pink'], alpha=0.8, edgecolor='black')
    ax5.set_xlabel("Subset", fontsize=11, fontweight='bold', labelpad=8)
    ax5.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax5.set_title("Gender Distribution Across Subsets", fontsize=11, fontweight='bold', pad=10)
    ax5.legend(title='Gender', fontsize=9, loc='upper right')
    ax5.tick_params(axis='x', rotation=30, labelsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Hand × Is_Altered Cross-tabulation
    ax6 = fig.add_subplot(gs[1, 2])
    crosstab2 = pd.crosstab(df['hand'], df['is_altered'])
    crosstab2.plot(kind='bar', ax=ax6, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax6.set_xlabel("Hand", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax6.set_title("Hand Distribution: Real vs Altered", fontsize=11, fontweight='bold', pad=10)
    ax6.legend(title='Is Altered', labels=['Real (0)', 'Altered (1)'], fontsize=9, loc='upper right')
    ax6.tick_params(axis='x', rotation=0)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Subjects per Subset
    ax7 = fig.add_subplot(gs[2, 0])
    subjects_per_subset = df.groupby('subset')['subject_id'].nunique()
    subset_order = ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
    subjects_per_subset = subjects_per_subset.reindex([s for s in subset_order if s in subjects_per_subset.index])
    bars = ax7.bar(range(len(subjects_per_subset)), subjects_per_subset.values, 
                   color=['steelblue', 'lightgreen', 'orange', 'crimson'], alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax7.set_xlabel("Subset", fontsize=11, fontweight='bold', labelpad=8)
    ax7.set_ylabel("Unique Subjects", fontsize=11, fontweight='bold')
    ax7.set_title("Unique Subjects per Subset", fontsize=11, fontweight='bold', pad=10)
    ax7.set_xticks(range(len(subjects_per_subset)))
    ax7.set_xticklabels(subjects_per_subset.index, rotation=30, ha='right', fontsize=9)
    ax7.set_ylim(0, subjects_per_subset.max() * 1.1)
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Samples per Gender per Subset (Stacked)
    ax8 = fig.add_subplot(gs[2, 1])
    gender_subset = df.groupby(['subset', 'gender']).size().unstack(fill_value=0)
    subset_order = ['Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard']
    gender_subset = gender_subset.reindex([s for s in subset_order if s in gender_subset.index])
    gender_subset.plot(kind='bar', stacked=True, ax=ax8, color=['skyblue', 'pink'], alpha=0.8, edgecolor='black')
    ax8.set_xlabel("Subset", fontsize=11, fontweight='bold', labelpad=8)
    ax8.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax8.set_title("Stacked: Gender per Subset", fontsize=11, fontweight='bold', pad=10)
    ax8.legend(title='Gender', fontsize=9, loc='upper right')
    ax8.tick_params(axis='x', rotation=30, labelsize=9)
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')
    
    summary_data = [
        ["Total Samples", f"{len(df):,}"],
        ["Total Subjects", f"{df['subject_id'].nunique():,}"],
        ["Real Samples", f"{(df['is_altered']==0).sum():,}"],
        ["Altered Samples", f"{(df['is_altered']==1).sum():,}"],
        ["Male Samples", f"{(df['gender']=='M').sum():,}"],
        ["Female Samples", f"{(df['gender']=='F').sum():,}"],
        ["Left Hand", f"{(df['hand']=='Left').sum():,}"],
        ["Right Hand", f"{(df['hand']=='Right').sum():,}"],
        ["Avg Samples/Subject", f"{subject_counts.mean():.1f}"],
        ["Subsets", f"{df['subset'].nunique()}"],
    ]
    
    table = ax9.table(cellText=summary_data, colLabels=["Metric", "Value"],
                     cellLoc='left', loc='upper center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    ax9.set_title("Dataset Summary Statistics", fontsize=11, fontweight='bold', pad=15)
    
    # Overall title
    fig.suptitle("Comprehensive Exploratory Data Analysis (EDA) - SOCOFing Dataset", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    utils.save_fig(f"{config.REPORTS_DIR}/comprehensive_eda.png")
    print(f"✅ Comprehensive EDA dashboard saved to {config.REPORTS_DIR}/comprehensive_eda.png")
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Total unique subjects: {df['subject_id'].nunique():,}")
    print(f"Average samples per subject: {subject_counts.mean():.2f}")
    print(f"Min samples per subject: {subject_counts.min()}")
    print(f"Max samples per subject: {subject_counts.max()}")
    print(f"\nGender breakdown:")
    for gender, count in df['gender'].value_counts().items():
        print(f"  {gender}: {count:,} ({count/len(df)*100:.2f}%)")
    print(f"\nHand breakdown:")
    for hand, count in df['hand'].value_counts().items():
        print(f"  {hand}: {count:,} ({count/len(df)*100:.2f}%)")


def main():
    """Run all visualizations."""
    utils.ensure_dirs()
    
    viz_preprocessing()
    viz_class_balance()
    viz_subset_distribution()
    viz_comprehensive_eda()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
