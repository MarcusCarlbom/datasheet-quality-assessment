import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
likes_df = pd.read_excel('LikesHuggingface.xlsx')
quality_df = pd.read_csv('quality_summary.csv')

# Filter only HuggingFace datasets
hf_quality = quality_df[quality_df['source'] == 'huggingface'].copy()

# Clean dataset names from Excel
def clean_dataset_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip().rstrip('"')
    parts = name.split('/')
    return parts[-1]

likes_df['cleaned_name'] = likes_df['datasets'].apply(clean_dataset_name)

# Manual matching for datasets with parent/child relationships
matching_map = {}

for _, quality_row in hf_quality.iterrows():
    quality_name = quality_row['dataset']
    
    # Try exact match first
    if quality_name in likes_df['cleaned_name'].values:
        match = likes_df[likes_df['cleaned_name'] == quality_name]['Likes'].values[0]
        matching_map[quality_name] = match
    else:
        # Try partial matching
        for _, likes_row in likes_df.iterrows():
            likes_name = likes_row['cleaned_name']
            # Check if quality name starts with likes name or contains it
            if quality_name.startswith(likes_name) or likes_name in quality_name:
                matching_map[quality_name] = likes_row['Likes']
                break

# Create merged dataframe
merged_data = []
for _, row in hf_quality.iterrows():
    dataset_name = row['dataset']
    if dataset_name in matching_map:
        row_dict = row.to_dict()
        row_dict['Likes'] = matching_map[dataset_name]
        merged_data.append(row_dict)

merged_df = pd.DataFrame(merged_data)

# Count total issues
issue_columns = [
    'has_duplicate_rows', 'has_empty_rows', 'excessive_missing_cols',
    'mixed_types_cols', 'constant_features', 'duplicate_columns',
    'has_ambiguous_labels', 'has_train_test_contamination'
]

merged_df['total_issues'] = merged_df[issue_columns].sum(axis=1)
# Average their issues and use one representative name
grouped_data = []
for likes_value in merged_df['Likes'].unique():
    subset = merged_df[merged_df['Likes'] == likes_value]
    
    # Get base name (remove suffixes and prefixes)
    base_names = []
    for name in subset['dataset']:
        base = name.split('_')[0]
        if 'glue' in name.lower():
            base = 'glue'
        elif 'tweet_eval' in name.lower():
            base = 'tweet_eval'
        base_names.append(base)
    
    # Use most common base name or first one
    from collections import Counter
    most_common = Counter(base_names).most_common(1)[0][0]
    
    # If multiple datasets, show count
    if len(subset) > 1:
        display_name = f"{most_common} ({len(subset)} variants)"
    else:
        display_name = subset['dataset'].values[0]
    
    grouped_data.append({
        'name': display_name,
        'likes': likes_value,
        'avg_issues': subset['total_issues'].mean(),
        'max_issues': subset['total_issues'].max(),
        'min_issues': subset['total_issues'].min(),
        'count': len(subset)
    })

grouped_df = pd.DataFrame(grouped_data)

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(18, 12))

# Use max issues for color (worst case scenario)
colors = plt.cm.RdYlGn_r(grouped_df['max_issues'] / max(grouped_df['max_issues'].max(), 1))

# Size based on number of variants
sizes = [300 + (count-1)*100 for count in grouped_df['count']]

scatter = ax.scatter(grouped_df['avg_issues'], grouped_df['likes'], 
                     s=sizes, alpha=0.7, c=colors, edgecolors='black', linewidth=2.5)

for idx, row in grouped_df.iterrows():
    ax.annotate(row['name'], 
                (row['avg_issues'], row['likes']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=13, alpha=0.9, fontweight='bold')

ax.set_xlabel('Average Quality Issues', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of Likes', fontsize=20, fontweight='bold')
ax.set_title('HuggingFace Datasets: Popularity vs Quality Issues', fontsize=24, fontweight='bold', pad=25)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Add correlation info with larger font
correlation = grouped_df['likes'].corr(grouped_df['avg_issues'])
info_text = f'Correlation: {correlation:.3f}\nDataset Groups: {len(grouped_df)}\nTotal Datasets: {len(merged_df)}'
ax.text(0.02, 0.98, info_text, 
        transform=ax.transAxes, fontsize=15, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Add note about point sizes
ax.text(0.98, 0.02, 'Larger points = multiple dataset variants', 
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
        horizontalalignment='right', style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('huggingface_likes_vs_quality.png', dpi=300, bbox_inches='tight')

print("Done!")