import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from itertools import combinations
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import random
from pathlib import Path
from config import settings
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Template explanations dictionary
TEMPLATE_EXPLANATIONS = {
        'Response': "After activity {A}, {B} MUST eventually occur",
        'Precedence': "{B} can ONLY happen if {A} occurred before",
        'Not Chain Response': "{A} is NEVER immediately followed by {B}",
        'Alternate Response': "Every {A} must be directly followed by {B}, with no other {A} in between",
        'Absence': "Activity {A} MUST NOT occur at all",
        'Existence': "Activity {A} MUST occur at least once",
        'Not Chain Precedence': "{B} is NEVER immediately preceded by {A}",
        'Not Precedence': "{B} can NEVER occur if {A} happened before",
        'Not Response': "After activity {A}, {B} MUST NEVER occur",
        'Choice': "Either activity {A} or activity {B} (or both) MUST occur",
        'End': "The process MUST end with activity {A}",
        'Init': "Activity {A} MUST be the first activity",
        'Alternate Precedence': "Every {B} must be directly preceded by {A}, with no other {B} in between",
        'Chain Precedence': "Activity {B} MUST be immediately preceded by {A}",
        'Chain Response': "Activity {B} MUST immediately follow {A}",
        'Responded Existence': "If {A} occurs, then {B} MUST occur at least once afterward"
    }

class CommunityComparator:
    def __init__(self, temporal_results_path: str):
        self.df = self.load_constraints(temporal_results_path)
        self.distance_matrices = self.compute_histogram_distance_matrix()
        
    def load_constraints(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess constraints data"""
        df = pd.read_csv(file_path)
        df['constraint_id'] = df.apply(
            lambda row: f"{row['constraint_type']}({row['A']}{f',{row['B']}' if pd.notna(row['B']) else ''})",
            axis=1
        )
        
        # Parse histogram bins/counts into lists
        df['duration_histogram_bins'] = df['duration_histogram_bins'].apply(
            lambda x: eval(x) if pd.notna(x) and len(str(x).strip()) > 0 else []
        )
        df['duration_histogram_counts'] = df['duration_histogram_counts'].apply(
            lambda x: eval(x) if pd.notna(x) and len(str(x).strip()) > 0 else []
        )
        return df
    
    def compute_histogram_distance_matrix(self) -> Dict[str, pd.DataFrame]:
        """Compute Wasserstein distance matrices for all constraints"""
        communities = sorted(self.df['cluster_id'].unique())
        constraint_ids = self.df['constraint_id'].unique()
        distance_matrices = {}
        
        for constraint_id in constraint_ids:
            sub_df = self.df[self.df['constraint_id'] == constraint_id]
            
            if sub_df['duration_histogram_bins'].apply(len).sum() == 0:
                continue

            distance_matrix = pd.DataFrame(index=communities, columns=communities, dtype=float)
            
            for c1, c2 in combinations(communities, 2):
                h1 = sub_df[sub_df['cluster_id'] == c1]
                h2 = sub_df[sub_df['cluster_id'] == c2]

                if h1.empty or h2.empty:
                    distance = 1.0
                else:
                    bins1 = h1['duration_histogram_bins'].values[0]
                    counts1 = h1['duration_histogram_counts'].values[0]
                    bins2 = h2['duration_histogram_bins'].values[0]
                    counts2 = h2['duration_histogram_counts'].values[0]

                    if bins1 != bins2 or len(counts1) != len(counts2) or len(counts1) == 0:
                        distance = 1.0
                    else:
                        p1 = np.array(counts1) / np.sum(counts1) if np.sum(counts1) > 0 else np.zeros(len(counts1))
                        p2 = np.array(counts2) / np.sum(counts2) if np.sum(counts2) > 0 else np.zeros(len(counts2))
                        distance = wasserstein_distance(
                            u_values=bins1[:-1], 
                            v_values=bins2[:-1], 
                            u_weights=p1, 
                            v_weights=p2
                        )

                distance_matrix.at[c1, c2] = distance
                distance_matrix.at[c2, c1] = distance

            for c in communities:
                distance_matrix.at[c, c] = 0.0
            
            distance_matrices[constraint_id] = distance_matrix.fillna(1.0)
        
        return distance_matrices
    
    def plot_community_heatmap(self, community_id: int) -> Optional[plt.Figure]:
        """Generate heatmap for a specific community"""
        heatmap_data = []
        raw_data = []
        constraint_names = []
        all_other_communities = set()

        for constraint_id, matrix in self.distance_matrices.items():
            if community_id not in matrix.index:
                continue

            distances = matrix.loc[community_id].drop(community_id)
            raw_row = distances.values
            raw_data.append(raw_row)
            all_other_communities.update(distances.index)

            # Shorten constraint name for display
            template, activities = constraint_id.split('(')
            activities = activities.rstrip(')')
            if ',' in activities:
                a, b = activities.split(',')
                short_activities = f"{self.shorten_activity_name(a)},{self.shorten_activity_name(b)}"
            else:
                short_activities = self.shorten_activity_name(activities)
            constraint_names.append(f"{template}({short_activities})")

        if not raw_data:
            logger.warning(f"No constraints found for community {community_id}")
            return None

        # Normalize values
        all_values = [val for sublist in raw_data for val in sublist]
        min_val, max_val = min(all_values), max(all_values)
        if max_val == min_val:
            logger.warning("All distances are the same. Cannot normalize.")
            return None

        norm_data = []
        for row in raw_data:
            norm_row = [(v - min_val) / (max_val - min_val) for v in row]
            norm_data.append(norm_row)

        # Build DataFrame
        other_communities = sorted(list(all_other_communities))
        heatmap_df = pd.DataFrame(norm_data, index=constraint_names, columns=other_communities)
        heatmap_df = heatmap_df.mask(heatmap_df < settings.MIN_DISTANCE_THRESHOLD)
        heatmap_df = heatmap_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        if heatmap_df.empty:
            logger.warning(f"No data remaining after applying min_distance threshold for community {community_id}")
            return None

        # Sort data
        heatmap_df = heatmap_df.loc[heatmap_df.max(axis=1).sort_values(ascending=False).index]
        heatmap_df = heatmap_df.loc[:, heatmap_df.mean().sort_values().index]

        # Save constraint list
        output_dir = Path(settings.HEATMAP_OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        y_vector_file = output_dir / f"community_{community_id}_constraints.txt"
        with open(y_vector_file, 'w') as f:
            f.write("\n".join(heatmap_df.index))

        # Plot
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            heatmap_df,
            cmap="YlOrRd",
            annot=False,
            linewidths=0.5,
            cbar_kws={'label': 'Normalized Wasserstein Distance (0–1)'},
            vmin=settings.MIN_DISTANCE_THRESHOLD,
            vmax=1.0
        )

        plt.title(f"Community {community_id} vs Other Communities\n(Constraints with normalized distance > {settings.MIN_DISTANCE_THRESHOLD})")
        plt.xlabel("Other Communities", fontsize=16)
        plt.ylabel("Constraints", fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(15)
        plt.tight_layout()
        
        return ax.figure
    
    def compare_communities(
    self,
    community1_id: int,
    community2_id: int,
    random_seed: Optional[int] = None) -> Tuple[str, Dict[str, float]]:
        """
        Compare two communities and return explanation text and heatmap scores
        """
        if random_seed is not None:
            random.seed(random_seed)

        # First collect all distances for normalization
        all_distances = []
        for matrix in self.distance_matrices.values():
            if community1_id in matrix.index:
                distances = matrix.loc[community1_id].drop(community1_id).values
                all_distances.extend(distances)
        
        # Calculate normalization parameters
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            normalize = max_dist > min_dist
        else:
            normalize = False

        # Identify shared constraints
        comm1_constraints = set(self.df[self.df['cluster_id'] == community1_id]['constraint_id'])
        comm2_constraints = set(self.df[self.df['cluster_id'] == community2_id]['constraint_id'])
        shared_constraints = comm1_constraints & comm2_constraints

        differences = []
        heatmap_scores = {}
        
        for constraint_id in shared_constraints:
            matrix = self.distance_matrices.get(constraint_id)
            if matrix is None or community1_id not in matrix.index or community2_id not in matrix.columns:
                continue

            # Get raw distance
            distance = matrix.at[community1_id, community2_id]
            
            # Normalize the distance
            if normalize:
                normalized_distance = (distance - min_dist) / (max_dist - min_dist)
            else:
                normalized_distance = 0.5  # default if no variation
                
            heatmap_scores[constraint_id] = normalized_distance

            comm1_data = self.df[
                (self.df['constraint_id'] == constraint_id) &
                (self.df['cluster_id'] == community1_id)
            ]
            comm2_data = self.df[
                (self.df['constraint_id'] == constraint_id) &
                (self.df['cluster_id'] == community2_id)
            ]

            template = constraint_id.split('(')[0]
            activities = constraint_id.split('(')[1].rstrip(')').split(',')
            A = activities[0]
            B = activities[1] if len(activities) > 1 else None

            differences.append((distance, normalized_distance, template, A, B, constraint_id, comm1_data, comm2_data))

        # Randomly sample differences
        random.shuffle(differences)
        selected_differences = differences[:min(settings.NUM_COMPARISON_CONSTRAINTS, len(differences))]
        selected_differences.sort(reverse=True, key=lambda x: x[0])  # Sort by raw distance

        # Generate explanation
        explanation = [
            f"\n=== Comparison between Community {community1_id} and Community {community2_id} ===",
            f"Random {settings.NUM_COMPARISON_CONSTRAINTS} differing constraints:\n"
        ]

        for diff in selected_differences:
            _, normalized_distance, template, A, B, constraint_id, comm1_data, comm2_data = diff
            base_explanation = TEMPLATE_EXPLANATIONS.get(template, f"The constraint {template} between {A} and {B} differs significantly")
            formatted_explanation = base_explanation.format(A=A, B=B) if B else base_explanation.format(A=A)

            # Duration comparison
            duration_comp = ""
            if not comm1_data.empty and not comm2_data.empty and B is not None:
                dur1 = comm1_data['duration_mean_mins'].values[0] / 60.0
                dur2 = comm2_data['duration_mean_mins'].values[0] / 60.0

                if not (np.isnan(dur1) or np.isnan(dur2)) and abs(dur1 - dur2) > settings.DURATION_THRESHOLD_HOURS:
                    direction = "longer" if dur1 > dur2 else "shorter"
                    duration_comp = f"\n  • Duration between {A} and {B}: {dur1:.1f}h vs {dur2:.1f}h (Community {community1_id} is {direction})"

            # Behavioral note
            comment = ""
            if normalized_distance > 0.7:
                if dur1 < dur2:
                    comment = f"\n  - Behavioral note: Constraint is stronger in Community {community1_id} vs Community {community2_id}\n"
                else:
                    comment = f"\n  - Behavioral note: Constraint is stronger in Community {community2_id} vs Community {community1_id}\n"
            else:
                if dur1 < dur2:
                    comment = f"\n  - Behavioral note: Constraint is moderately different in Community {community1_id} vs Community {community2_id}\n"
                else:
                    comment = f"\n  - Behavioral note: Constraint is moderately different in Community {community2_id} vs Community {community1_id}\n"

            explanation.append(
                f"{constraint_id} (distance: {normalized_distance:.2f}):\n"
                f"  - Meaning: {formatted_explanation}"
                f"{duration_comp}"
                f"{comment}"
            )

        return '\n'.join(explanation), heatmap_scores

    @staticmethod
    def shorten_activity_name(name: str, max_length: int = 10) -> str:
        """Shorten activity names for display"""
        if len(name) <= max_length:
            return name
        part_length = max_length // 3
        return f"{name[:part_length]}...{name[-part_length:]}"
    
    @staticmethod
    def label_time(hour: float) -> str:
        """Convert hour to time of day label"""
        return (
            "early morning" if hour < 6 else
            "morning" if hour < 12 else
            "afternoon" if hour < 18 else
            "evening"
        )
    
    @staticmethod
    def save_colored_explanation(
        explanation: str,
        output_path: str,
        heatmap_scores: Dict[str, float]
    ) -> None:
        """Save explanation as a colored image"""
        try:
            cmap = cm.get_cmap('YlOrRd')
            norm = mcolors.Normalize(vmin=0, vmax=1)
            lines = explanation.split('\n')
            
            fig_height = max(6, len(lines) * 0.3 + 0.5)
            fig = plt.figure(figsize=(10, fig_height))
            ax_text = fig.add_axes([0.05, 0.1, 0.9, 0.85])
            
            y_position = 0.98
            line_height = 0.04
            current_score = 0.5
            
            for line in lines:
                if not line.strip():
                    y_position -= line_height * 0.5
                    continue
                  
                for constraint_id in heatmap_scores:
                    if constraint_id in line:
                        current_score = heatmap_scores[constraint_id]
                        break
                
                color = 'black'
                fontweight = 'normal'
                fontsize = 11
                bbox = None
                
                if line.startswith('==='):
                    fontweight = 'bold'
                    fontsize = 14
                elif line.strip().endswith('):'):
                    fontweight = 'bold'
                    fontsize = 12
                elif line.strip().startswith('- Behavioral note'):
                    highlight_color = cmap(norm(current_score))
                    bbox = {
                        'facecolor': highlight_color,
                        'alpha': 0.4,
                        'pad': 6,
                        'boxstyle': 'round,pad=0.5',
                        'edgecolor': highlight_color
                    }
                
                ax_text.text(
                    0.05, y_position, line, 
                    color=color,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    transform=ax_text.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=bbox
                )
                y_position -= line_height
                
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(
                sm, 
                ax=ax_text,
                orientation='horizontal',
                fraction=0.05,
                pad=0.05,
                aspect=40
            )
            cbar.set_label('Behavioral Difference Strength (0 = weak, 1 = strong)', fontsize=10)
            ax_text.axis('off')
            
            output_dir = Path(output_path).parent
            output_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved explanation image to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise