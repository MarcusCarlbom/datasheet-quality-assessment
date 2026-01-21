"""
visualize_results.py
Visualize data quality validation results from CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_repository_comparison(csv_path: str):
    """
    Visualize repository_comparison.csv
    Shows which repositories have which problems
    """
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Repository Quality Comparison', fontsize=16, fontweight='bold')
    
    # 1. Critical Issues Overview (TOP LEFT)
    ax1 = axes[0, 0]
    critical_cols = ['contamination_pct', 'ambiguous_labels_pct', 
                     'constant_features_pct', 'excessive_missing_pct']
    critical_data = df[['repository'] + critical_cols].set_index('repository')
    
    critical_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Critical Issues (% of datasets affected)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Datasets (%)')
    ax1.set_xlabel('Repository')
    ax1.legend(['Contamination', 'Ambiguous Labels', 'Constant Features', 'Excessive Missing'],
               loc='upper left', fontsize=9)
    ax1.set_xticklabels(df['repository'], rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Data Quality Metrics (TOP RIGHT)
    ax2 = axes[0, 1]
    quality_cols = ['high_duplicates_pct', 'mixed_types_pct', 'duplicate_columns_pct']
    quality_data = df[['repository'] + quality_cols].set_index('repository')
    
    quality_data.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Data Quality Issues (% of datasets affected)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage of Datasets (%)')
    ax2.set_xlabel('Repository')
    ax2.legend(['High Duplicates', 'Mixed Types', 'Duplicate Columns'],
               loc='upper left', fontsize=9)
    ax2.set_xticklabels(df['repository'], rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Representation Issues (BOTTOM LEFT)
    ax3 = axes[1, 0]
    repr_cols = ['enum_as_numeric_pct', 'unnormalized_pct']
    repr_data = df[['repository'] + repr_cols].set_index('repository')
    
    repr_data.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('Representation Issues (% of datasets affected)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage of Datasets (%)')
    ax3.set_xlabel('Repository')
    ax3.legend(['Enum as Numeric', 'Unnormalized Features'],
               loc='upper left', fontsize=9)
    ax3.set_xticklabels(df['repository'], rotation=0)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Average Rates (BOTTOM RIGHT)
    ax4 = axes[1, 1]
    rate_cols = ['avg_missing_rate_pct', 'avg_duplicate_rate_pct']
    rate_data = df[['repository'] + rate_cols].set_index('repository')
    
    rate_data.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Average Data Rates', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xlabel('Repository')
    ax4.legend(['Missing Rate', 'Duplicate Rate'],
               loc='upper left', fontsize=9)
    ax4.set_xticklabels(df['repository'], rotation=0)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('repository_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: repository_comparison.png")
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("REPOSITORY COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("\n")


def visualize_dataset_details(csv_path: str):
    """
    Visualize quality_summary_detailed.csv
    Shows per-dataset quality metrics
    """
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Quality Details', fontsize=16, fontweight='bold')
    
    # 1. Contamination by Dataset (TOP LEFT)
    ax1 = axes[0, 0]
    contaminated = df[df['contamination_pct'] > 0].sort_values('contamination_pct', ascending=False)
    
    if len(contaminated) > 0:
        # Color code: red >20%, orange 5-20%, yellow 1-5%, green <1%
        colors = ['#d62728' if x >= 20 else '#ff7f0e' if x >= 5 else '#ffd700' if x >= 1 else '#2ca02c' 
                  for x in contaminated['contamination_pct']]
        
        # Limit to top 20 worst cases
        contaminated_display = contaminated.head(20)
        colors_display = colors[:20]
        
        ax1.barh(contaminated_display['dataset'], contaminated_display['contamination_pct'], color=colors_display)
        ax1.set_xlabel('Contamination (%)')
        ax1.set_title(f'Train/Test Contamination by Dataset (Top {len(contaminated_display)} of {len(contaminated)})', 
                     fontsize=12, fontweight='bold')
        ax1.axvline(x=1, color='#ffd700', linestyle='--', alpha=0.5, label='1% threshold')
        ax1.axvline(x=5, color='#ff7f0e', linestyle='--', alpha=0.5, label='5% threshold')
        ax1.axvline(x=20, color='#d62728', linestyle='--', alpha=0.5, label='20% SEVERE')
        ax1.legend(fontsize=8, loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No contamination detected!', 
                ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
    
    # 2. Error/Warning/Info Distribution (TOP RIGHT)
    ax2 = axes[0, 1]
    severity_cols = ['errors', 'warnings', 'infos']
    severity_data = df.groupby('source')[severity_cols].sum()
    
    severity_data.plot(kind='bar', stacked=True, ax=ax2, 
                      color=['#d62728', '#ff7f0e', '#1f77b4'], width=0.6)
    ax2.set_title('Issue Severity by Repository', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Repository')
    ax2.legend(['Errors', 'Warnings', 'Info'], loc='upper left', fontsize=9)
    ax2.set_xticklabels(severity_data.index, rotation=0)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Missing & Duplicate Rates (BOTTOM LEFT)
    ax3 = axes[1, 0]
    
    # Create scatter plot
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        ax3.scatter(source_df['missing_rate_pct'], source_df['duplicate_rate_pct'], 
                   label=source, s=100, alpha=0.6)
    
    ax3.set_xlabel('Missing Rate (%)', fontsize=10)
    ax3.set_ylabel('Duplicate Rate (%)', fontsize=10)
    ax3.set_title('Missing vs Duplicate Rates', fontsize=12, fontweight='bold')
    ax3.legend(title='Repository', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # 4. Feature Issues Heatmap (BOTTOM RIGHT)
    ax4 = axes[1, 1]
    
    # Select feature issue columns
    feature_cols = ['enum_as_numeric_features', 'unnormalized_features', 
                   'constant_features', 'mixed_types_columns']
    heatmap_data = df[['dataset', 'source'] + feature_cols].copy()
    
    # Only show datasets with issues
    heatmap_data['total_issues'] = heatmap_data[feature_cols].sum(axis=1)
    heatmap_data = heatmap_data[heatmap_data['total_issues'] > 0]
    
    if len(heatmap_data) > 0:
        # Limit to top 15 datasets with most issues for readability
        heatmap_data = heatmap_data.nlargest(15, 'total_issues')
        heatmap_display = heatmap_data[feature_cols].set_index(heatmap_data['dataset']).T
        
        sns.heatmap(heatmap_display, annot=True, fmt='g', cmap='YlOrRd', 
                   cbar_kws={'label': 'Count'}, ax=ax4, linewidths=0.5)
        ax4.set_title(f'Feature Issues (Top {len(heatmap_data)} Datasets)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Issue Type')
    else:
        ax4.text(0.5, 0.5, 'No feature issues detected!', 
                ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_details.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dataset_details.png")
    plt.show()

def create_summary_dashboard(repo_csv: str, details_csv: str):
    """
    Create a comprehensive single-page dashboard
    """
    repo_df = pd.read_csv(repo_csv)
    details_df = pd.read_csv(details_csv)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Data Quality Validation Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Repository Overview (TOP LEFT - spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    critical_issues = repo_df[['repository', 'contamination_count', 
                               'constant_features_count', 'excessive_missing_count']].set_index('repository')
    critical_issues.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Critical Issues Count by Repository', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Datasets')
    ax1.legend(['Contamination', 'Constant Features', 'Excessive Missing'], fontsize=9)
    ax1.set_xticklabels(repo_df['repository'], rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Dataset Count (TOP RIGHT)
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax2.pie(repo_df['num_datasets'], labels=repo_df['repository'], autopct='%1.0f%%',
           colors=colors, startangle=90)
    ax2.set_title('Dataset Distribution', fontsize=12, fontweight='bold')
    
    # 3. Contamination Timeline (MIDDLE LEFT - spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    
    contaminated = details_df[details_df['contamination_pct'] > 0].sort_values('contamination_pct')
    if len(contaminated) > 0:
        # Show top 15 worst cases
        contaminated_display = contaminated.tail(15)
        
        colors_cont = contaminated_display['source'].map({'huggingface': '#1f77b4', 
                                                   'openml': '#ff7f0e', 
                                                   'uci': '#2ca02c'})
        ax3.barh(range(len(contaminated_display)), contaminated_display['contamination_pct'], color=colors_cont)
        ax3.set_yticks(range(len(contaminated_display)))
        ax3.set_yticklabels(contaminated_display['dataset'], fontsize=9)
        ax3.set_xlabel('Contamination (%)')
        ax3.set_title(f'Train/Test Contamination (Top {len(contaminated_display)} of {len(contaminated)} affected)', 
                     fontsize=12, fontweight='bold')
        ax3.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='1%')
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='5%')
        ax3.axvline(x=20, color='darkred', linestyle='--', alpha=0.5, linewidth=2, label='20% SEVERE')
        ax3.legend(fontsize=8, loc='lower right')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '✓ No Contamination Detected', 
                ha='center', va='center', fontsize=16, color='green', fontweight='bold')
        ax3.axis('off')
    
    # 4. Issue Severity Summary (MIDDLE RIGHT)
    ax4 = fig.add_subplot(gs[1, 2])
    
    total_issues = pd.DataFrame({
        'Severity': ['Errors', 'Warnings', 'Info'],
        'Count': [details_df['errors'].sum(), 
                 details_df['warnings'].sum(), 
                 details_df['infos'].sum()]
    })
    
    colors_sev = ['#d62728', '#ff7f0e', '#1f77b4']
    ax4.bar(total_issues['Severity'], total_issues['Count'], color=colors_sev)
    ax4.set_title('Total Issues by Severity', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(total_issues['Count']):
        ax4.text(i, v + 0.5, str(int(v)), ha='center', fontweight='bold')
    
    # 5. Feature Issues by Repository (BOTTOM LEFT)
    ax5 = fig.add_subplot(gs[2, 0])
    
    feature_issues = details_df.groupby('source')[['enum_as_numeric_features', 
                                                    'unnormalized_features']].sum()
    feature_issues.plot(kind='bar', ax=ax5, width=0.7)
    ax5.set_title('Representation Issues', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count')
    ax5.set_xlabel('Repository')
    ax5.legend(['Enum as Numeric', 'Unnormalized'], fontsize=8)
    ax5.set_xticklabels(feature_issues.index, rotation=0)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Data Quality Metrics (BOTTOM MIDDLE)
    ax6 = fig.add_subplot(gs[2, 1])
    
    quality_metrics = repo_df[['repository', 'avg_missing_rate_pct', 
                               'avg_duplicate_rate_pct']].set_index('repository')
    quality_metrics.plot(kind='bar', ax=ax6, width=0.7)
    ax6.set_title('Average Data Quality', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Percentage (%)')
    ax6.set_xlabel('Repository')
    ax6.legend(['Missing Rate', 'Duplicate Rate'], fontsize=8)
    ax6.set_xticklabels(quality_metrics.index, rotation=0)
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Quality Score Card (BOTTOM RIGHT)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Calculate scores
    total_datasets = len(details_df)
    clean_datasets = len(details_df[(details_df['errors'] == 0) & 
                                   (details_df['contamination_pct'] == 0)])
    contaminated_datasets = (details_df['contamination_pct'] > 0).sum()
    
    scorecard_text = f"""
    QUALITY SCORECARD
    {'='*25}
    
    Total Datasets: {total_datasets}
    
    Clean Datasets: {clean_datasets}
    ({clean_datasets/total_datasets*100:.1f}%)
    
    Contaminated: {contaminated_datasets}
    ({contaminated_datasets/total_datasets*100:.1f}%)
    
    Total Errors: {details_df['errors'].sum()}
    Total Warnings: {details_df['warnings'].sum()}
    
    Best Repository:
    {repo_df.sort_values('contamination_count').iloc[0]['repository']}
    """
    
    ax7.text(0.1, 0.5, scorecard_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.savefig('quality_dashboard.png', dpi=300, bbox_inches='tight')
    print("Saved: quality_dashboard.png")
    plt.show()


def main():
    """
    Main function - run all visualizations
    """
    
    repo_csv = "repository_comparison.csv"
    details_csv = "quality_summary.csv"
    
    # Check files exist
    import os
    if not os.path.exists(repo_csv):
        print(f"Error: {repo_csv} not found!")
        print("Run data_validator.py first to generate the CSV files.")
        return
    
    if not os.path.exists(details_csv):
        print(f"Error: {details_csv} not found!")
        print("Run data_validator.py first to generate the CSV files.")
        return
    
    print("Generating visualizations...\n")
    
    visualize_repository_comparison(repo_csv)
    visualize_dataset_details(details_csv)
    create_summary_dashboard(repo_csv, details_csv)
    
    print("All visualizations complete!")


if __name__ == "__main__":
    main()