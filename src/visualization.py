import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import settings

def generate_heatmap(input_path, output_path):
    """Generate and save the TF-IDF heatmap visualization"""
    df = pd.read_csv(input_path)
    
    # Define global bin edges
    global_min = df['tfidf_score'].min()
    global_max = df['tfidf_score'].max()
    bin_edges = np.linspace(global_min, global_max, num=settings.HEATMAP_BINS + 1)
    
    # Compute histogram for each community
    community_ids = sorted(df['community_id'].unique())
    heatmap_data = []
    
    for community_id in community_ids:
        community_data = df[df['community_id'] == community_id]
        freq, _ = np.histogram(community_data['tfidf_score'], bins=bin_edges)
        heatmap_data.append(freq)
    
    # Create DataFrame for Seaborn
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=community_ids,
        columns=[f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]
    )
    
    # Generate plot
    plt.figure(figsize=(15, 10))
    heatmap = sns.heatmap(
        heatmap_df,
        cmap=settings.HEATMAP_COLORMAP,
        annot=False,
        linewidths=0.5,
        cbar_kws={'label': 'Frequency of Constraints'}
    )
    
    # Adjust colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Frequency of Constraints', size=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Label adjustments
    plt.xlabel("Range of TF-IDF Scores", fontsize=14)
    plt.ylabel("Community ID", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")
