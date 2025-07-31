import os
from pathlib import Path
from config import settings
from src.community_analyzer import analyze_communities
from src.constraint_ranker import rank_constraints
from src.visualization import generate_heatmap
from src.temporal_analyzer import analyze_temporal_constraints
from src.community_comparator import CommunityComparator


def ensure_directory_exists(path):
    """Ensure output directory exists"""
    os.makedirs(path, exist_ok=True)

def main():
    # Ensure results directory exists
    ensure_directory_exists(settings.RESULTS_DIR)
    
    print("Starting community analysis...")
    # Step 1: Run community analysis
    community_results_path = settings.RESULTS_DIR / settings.COMMUNITY_RESULTS_FILE
    analyze_communities(
        xes_path=settings.XES_FILE_PATH,
        community_csv_path=settings.COMMUNITY_CSV_PATH,
        output_dir=str(settings.RESULTS_DIR)
    )
    
    print("\nRanking constraints with TF-IDF...")
    # Step 2: Run TF-IDF ranking
    ranked_constraints_path = settings.RESULTS_DIR / settings.RANKED_CONSTRAINTS_FILE
    rank_constraints(
        input_path=str(community_results_path),
        output_path=str(ranked_constraints_path)
    )
    
    print("\nGenerating visualization...")
    # Step 3: Generate heatmap visualization
    heatmap_path = settings.RESULTS_DIR / settings.HEATMAP_FILE
    generate_heatmap(
        input_path=str(ranked_constraints_path),
        output_path=str(heatmap_path))
    
    # Step 4: Temporal analysis
    print("\nAnalyzing temporal constraints...")
    temporal_results_path = settings.RESULTS_DIR / "all_constraints_with_time.csv"
    analyze_temporal_constraints(
        xes_path=settings.XES_FILE_PATH,
        community_csv_path=settings.COMMUNITY_CSV_PATH,
        constraints_csv_path=str(community_results_path),
        output_path=str(temporal_results_path)
    )
    
    # Step 5: Community comparison
    print("\nSetting up community comparison...")
    comparator = CommunityComparator(str(temporal_results_path))
    
    # Generate heatmap for a specific community
    target_community = int(input("\nEnter community ID to generate heatmap: "))
    heatmap_fig = comparator.plot_community_heatmap(target_community)
    if heatmap_fig:
        heatmap_output_path = settings.RESULTS_DIR / settings.HEATMAP_OUTPUT_DIR / f"community_{target_community}_heatmap.png"
        ensure_directory_exists(heatmap_output_path.parent)
        heatmap_fig.savefig(str(heatmap_output_path), bbox_inches='tight', dpi=300)
        plt.close(heatmap_fig)
        print(f"Heatmap saved to {heatmap_output_path}")
    
    # Compare two communities
    print("\n=== Community Comparison ===")
    comm1 = int(input("Enter first community ID to compare: "))
    comm2 = int(input("Enter second community ID to compare: "))
    
    explanation, heatmap_scores = comparator.compare_communities(comm1, comm2, random_seed=42)
    
    # Save text explanation
    explanation_path = settings.RESULTS_DIR / settings.COMPARISON_OUTPUT_DIR / f"comparison_{comm1}_vs_{comm2}.txt"
    ensure_directory_exists(explanation_path.parent)
    with open(explanation_path, 'w') as f:
        f.write(explanation)
    print(f"\nText explanation saved to {explanation_path}")
    
    # Save visual explanation
    image_path = settings.RESULTS_DIR / settings.COMPARISON_OUTPUT_DIR / f"comparison_{comm1}_vs_{comm2}.png"
    CommunityComparator.save_colored_explanation(explanation, str(image_path), heatmap_scores)
    print(f"Visual explanation saved to {image_path}")
    
    
    print("\nAnalysis complete!")
    print(f"All results saved in: {settings.RESULTS_DIR}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()