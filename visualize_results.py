import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

REPO_COLORS = {
    'huggingface': '#FF9D00',
    'openml': '#4285F4', 
    'uci': '#34A853'
}

PROBLEM_TYPES = {
    'has_train_test_contamination': {
        'title': 'Train/Test Contamination',
        'color': "#DC113A"
    },
    'has_ambiguous_labels': {
        'title': 'Ambiguous Labels',
        'color': "#E95241"
    },
    'constant_features': {
        'title': 'Constant Features',
        'color': "#FF7F7F"
    },
    'excessive_missing_cols': {
        'title': 'Excessive Missing Values',
        'color': '#FF8C00'
    },
    'mixed_types_cols': {
        'title': 'Mixed Types',
        'color': '#FF8C00'
    },
    'has_duplicate_rows': {
        'title': 'Duplicate Rows',
        'color': '#FF8C00'
    },
    'has_empty_rows': {
        'title': 'Empty Rows',
        'color': "#FFCC02"
    },
    'duplicate_columns': {
        'title': 'Duplicate Columns',
        'color': "#00FF6E"
    },
    'number_as_string': {
        'title': 'Numbers as Strings',
        'color': "#0099FF"
    },
    'enum_as_numeric': {
        'title': 'Categorical as Integers',
        'color': "#0099FF"
    },
    'unnormalized_features': {
        'title': 'Unnormalized Features',
        'color': "#00A6FF"
    },
    'tailed_distributions': {
        'title': 'Skewed Distributions',
        'color': "#0099FF"
    }
}


def load_summary_data(filepath: str = "quality_summary.csv") -> tuple:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Cannot find {filepath}")
    
    df = pd.read_csv(filepath)
    
    contam_df = None
    contam_path = "contamination_details.csv"
    if Path(contam_path).exists():
        contam_df = pd.read_csv(contam_path)
    
    return df, contam_df


def create_critical_problem_viz(df: pd.DataFrame, problem_col: str, problem_info: dict, 
                                output_dir: str = "./visualizations", contam_df=None):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{problem_info['title']}", fontsize=16, fontweight='bold', color=problem_info['color'])
    
    is_boolean = df[problem_col].max() <= 1
    is_contamination = problem_col == 'has_train_test_contamination' and contam_df is not None
    
    ax1 = axes[0]
    if is_boolean:
        prevalence = df.groupby('source')[problem_col].mean() * 100
        total_affected = df.groupby('source')[problem_col].sum()
    else:
        prevalence = df.groupby('source')[problem_col].apply(lambda x: (x > 0).sum() / len(x) * 100)
        total_affected = df.groupby('source')[problem_col].apply(lambda x: (x > 0).sum())
    
    colors = [REPO_COLORS.get(repo, '#888888') for repo in prevalence.index]
    bars = ax1.bar(prevalence.index, prevalence.values, color=colors, width=0.6, alpha=0.8)
    ax1.set_ylabel('% of Datasets Affected', fontsize=11, fontweight='bold')
    ax1.set_title('Repository Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(prevalence.values) * 1.2 if prevalence.values.max() > 0 else 10)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, repo in zip(bars, prevalence.index):
        height = bar.get_height()
        count = int(total_affected[repo])
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({count})', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax2 = axes[1]

    if is_contamination:
        worst_contam = contam_df.nlargest(10, 'contamination_pct').copy()
        worst_contam['display_name'] = worst_contam['dataset']
        
        if len(worst_contam) > 0:
            colors = [REPO_COLORS.get(repo, '#888888') for repo in worst_contam['source']]
            y_pos = np.arange(len(worst_contam))
            
            ax2.barh(y_pos, worst_contam['contamination_pct'].values, color=colors, alpha=0.8)
            
            ax2.set_yticks(y_pos)
            labels = []
            for _, row in worst_contam.iterrows():
                label = f"{row['display_name'][:30]}"
                if len(row['display_name']) > 30:
                    label += "..."
                label += f" ({row['source']})"
                labels.append(label)
            ax2.set_yticklabels(labels, fontsize=9)
            ax2.set_xlabel('Contamination %', fontsize=11, fontweight='bold')
            ax2.set_title(f'Top {len(worst_contam)} Most Contaminated', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            
            ax2.axvline(x=0.5, color='#90EE90', linestyle='--', alpha=0.5, linewidth=1.5, label='0.5%')
            ax2.axvline(x=5, color='#FFD700', linestyle='--', alpha=0.5, linewidth=1.5, label='5%')
            ax2.axvline(x=20, color='#DC143C', linestyle='--', alpha=0.5, linewidth=1.5, label='20%')
            ax2.legend(fontsize=8, loc='lower right')
            
            for i, val in enumerate(worst_contam['contamination_pct'].values):
                ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '✓ No contamination found', 
                    ha='center', va='center', fontsize=14, color='green',
                    fontweight='bold', transform=ax2.transAxes)
            ax2.axis('off')
    else:
        if is_boolean:
            worst = df[df[problem_col] == 1].copy()
        else:
            worst = df[df[problem_col] > 0].copy()
        
        worst['display_name'] = worst['dataset'].str.replace('_train$', '', regex=True).str.replace('_test$', '', regex=True)
        
        if is_boolean:
            worst = worst.drop_duplicates(subset=['source', 'display_name'], keep='first')
        else:
            worst = worst.sort_values(problem_col, ascending=False).drop_duplicates(subset=['source', 'display_name'], keep='first')
        
        worst = worst.head(10)
        
        if len(worst) > 0:
            colors = [REPO_COLORS.get(repo, '#888888') for repo in worst['source']]
            y_pos = np.arange(len(worst))
            
            if is_boolean:
                ax2.barh(y_pos, [1]*len(worst), color=colors, alpha=0.8)
                ax2.set_xlim(0, 1.1)
                ax2.set_xticks([])
                ax2.spines['bottom'].set_visible(False)
                ax2.set_xlabel('')
            else:
                ax2.barh(y_pos, worst[problem_col].values, color=colors, alpha=0.8)
                ax2.set_xlabel('Count', fontsize=11, fontweight='bold')
                
                for i, val in enumerate(worst[problem_col].values):
                    ax2.text(val + (max(worst[problem_col].values) * 0.02), i, 
                            f'{int(val)}', va='center', fontsize=9, fontweight='bold')
            
            ax2.set_yticks(y_pos)
            labels = []
            for _, row in worst.iterrows():
                label = f"{row['display_name'][:35]}"
                if len(row['display_name']) > 35:
                    label += "..."
                label += f" ({row['source']})"
                labels.append(label)
            ax2.set_yticklabels(labels, fontsize=9)
            ax2.set_title(f'Top {len(worst)} Affected Datasets', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, '✓ No datasets affected', 
                    ha='center', va='center', fontsize=14, color='green',
                    fontweight='bold', transform=ax2.transAxes)
            ax2.axis('off')
    
    plt.tight_layout()
    
    safe_filename = problem_col.replace('_', '-')
    output_file = output_path / f"critical-{safe_filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_high_priority_viz(df: pd.DataFrame, problem_col: str, problem_info: dict,
                             output_dir: str = "./visualizations"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{problem_info['title']}", fontsize=14, fontweight='bold', color=problem_info['color'])
    
    is_boolean = df[problem_col].max() <= 1
    
    if is_boolean:
        prevalence = df.groupby('source')[problem_col].mean() * 100
        total_affected = df.groupby('source')[problem_col].sum()
    else:
        prevalence = df.groupby('source')[problem_col].apply(lambda x: (x > 0).sum() / len(x) * 100)
        total_affected = df.groupby('source')[problem_col].apply(lambda x: (x > 0).sum())
        avg_severity = df[df[problem_col] > 0].groupby('source')[problem_col].mean()
    
    colors = [REPO_COLORS.get(repo, '#888888') for repo in prevalence.index]
    bars = ax.bar(prevalence.index, prevalence.values, color=colors, width=0.6, alpha=0.8)
    
    ax.set_ylabel('% of Datasets Affected', fontsize=11, fontweight='bold')
    ax.set_xlabel('Repository', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(prevalence.values) * 1.25 if prevalence.values.max() > 0 else 10)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, repo in zip(bars, prevalence.index):
        height = bar.get_height()
        count = int(total_affected[repo])
        if is_boolean:
            label = f'{height:.1f}%\n({count} datasets)'
        else:
            avg = avg_severity.get(repo, 0)
            label = f'{height:.1f}%\n({count} datasets)\navg: {avg:.1f}'
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    safe_filename = problem_col.replace('_', '-')
    output_file = output_path / f"high-{safe_filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_overview_dashboard(df: pd.DataFrame, output_dir: str = "./visualizations"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Repository Quality Dashboard', fontsize=18, fontweight='bold')
    
    problem_cols = [col for col in PROBLEM_TYPES.keys() if col in df.columns]
    
    ax1 = fig.add_subplot(gs[0:2, 0])
    
    quality_scores = {}
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        has_problems = (source_data[problem_cols].sum(axis=1) > 0).sum()
        clean_pct = ((len(source_data) - has_problems) / len(source_data)) * 100
        quality_scores[source] = clean_pct
    
    colors = [REPO_COLORS.get(repo, '#888888') for repo in quality_scores.keys()]
    bars = ax1.barh(list(quality_scores.keys()), list(quality_scores.values()), color=colors, alpha=0.8)
    ax1.set_xlabel('% Clean Datasets', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Repository Quality', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, (repo, score) in zip(bars, quality_scores.items()):
        width = bar.get_width()
        total = len(df[df['source'] == repo])
        clean = int((score / 100) * total)
        ax1.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}% ({clean}/{total})', va='center', fontweight='bold', fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1:3])
    for source in df['source'].unique():
        source_data = df[df['source'] == source]
        ax2.scatter(source_data['rows'], source_data['columns'], 
                   label=source, alpha=0.5, s=50,
                   color=REPO_COLORS.get(source, '#888888'))
    ax2.set_xlabel('Rows (log)', fontsize=9)
    ax2.set_ylabel('Columns', fontsize=9)
    ax2.set_title('Dataset Sizes', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)
    
    ax3 = fig.add_subplot(gs[1, 1])
    avg_missing = df.groupby('source')['missing_rate_pct'].mean()
    colors = [REPO_COLORS.get(repo, '#888888') for repo in avg_missing.index]
    bars = ax3.bar(avg_missing.index, avg_missing.values, color=colors, width=0.6, alpha=0.8)
    ax3.set_ylabel('Missing Rate (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Average Missing Values', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(labelsize=9)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 2])
    avg_dup = df.groupby('source')['duplicate_rate_pct'].mean()
    colors = [REPO_COLORS.get(repo, '#888888') for repo in avg_dup.index]
    bars = ax4.bar(avg_dup.index, avg_dup.values, color=colors, width=0.6, alpha=0.8)
    ax4.set_ylabel('Duplicate Rate (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Average Duplicates', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(labelsize=9)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax5 = fig.add_subplot(gs[2, :])
    
    repo_prevalence = df.groupby('source').apply(
        lambda x: pd.Series({
            col: (x[col] > 0).sum() / len(x) * 100 if col in x.columns else 0
            for col in problem_cols
        }),
        include_groups=False
    )
    
    active_cols = [col for col in problem_cols if repo_prevalence[col].sum() > 0]
    heatmap_data = repo_prevalence[active_cols].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
               cbar_kws={'label': '% Affected'}, linewidths=0.5, ax=ax5,
               vmin=0, vmax=100)
    
    ax5.set_title('Problem Prevalence Across Repositories (%)', fontsize=11, fontweight='bold', pad=10)
    ax5.set_xlabel('Repository', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Problem Type', fontsize=10, fontweight='bold')
    
    ylabels = [PROBLEM_TYPES.get(col, {}).get('title', col.replace('_', ' ').title()) 
               for col in active_cols]
    ax5.set_yticklabels(ylabels, rotation=0, fontsize=8)
    ax5.tick_params(labelsize=9)

    output_file = output_path / "00-overview-dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    

def main():
    print("Generating visualizations...")
    
    df, contam_df = load_summary_data("quality_summary.csv")
    
    output_dir = "./visualizations"
    Path(output_dir).mkdir(exist_ok=True)
    
    create_overview_dashboard(df, output_dir)
    
    critical_problems = ['has_train_test_contamination', 'has_ambiguous_labels']
    critical_problems = [col for col in critical_problems if col in df.columns]

    for problem_col in critical_problems:
        create_critical_problem_viz(df, problem_col, PROBLEM_TYPES[problem_col], output_dir, contam_df)

    high_problems = ['constant_features', 'excessive_missing_cols', 'mixed_types_cols']
    high_problems = [col for col in high_problems if col in df.columns]
    
    for problem_col in high_problems:
        create_high_priority_viz(df, problem_col, PROBLEM_TYPES[problem_col], output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()