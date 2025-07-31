import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / 'src'
RESULTS_DIR = BASE_DIR / 'results'

# Input file paths
XES_FILE_PATH = r"E:\Sylvio\DECLARE\Prepaid_10.xes"
COMMUNITY_CSV_PATH = r"E:\Sylvio\DECLARE\Prepaid_10_k3_wa1_wt1_wtime0.5_modularity0.96_r1.0.csv"

# Analysis parameters
DECLARE_MIN_SUPPORT = 0.7
DECLARE_ITEMSETS_SUPPORT = 0.9
DECLARE_MAX_CARDINALITY = 1

# Output file names
COMMUNITY_RESULTS_FILE = "all_community_results.csv"
RANKED_CONSTRAINTS_FILE = "ranked_constraints_v2.csv"
HEATMAP_FILE = "tfidf_heatmap.png"

# Visualization settings
HEATMAP_BINS = 20
HEATMAP_COLORMAP = "icefire"


# Temporal Analysis Settings
DURATION_TEMPLATES = {
    'Response', 'Chain Response', 'Precedence', 'Chain Precedence',
    'Alternate Response', 'Alternate Precedence', 'Responded Existence',
    'Not Chain Response', 'Not Precedence', 'Not Response', 'Not Chain Precedence'
}

TIMEWINDOW_TEMPLATES = {
    'Init', 'End', 'Existence', 'Absence', 'Choice', 'Exactly',
    'Response', 'Chain Response', 'Precedence', 'Chain Precedence',
    'Alternate Response', 'Alternate Precedence', 'Responded Existence',
    'Not Chain Response', 'Not Precedence', 'Not Response', 'Not Chain Precedence'
}

# Community Comparison Settings
COMPARISON_OUTPUT_DIR = "community_comparisons"
HEATMAP_OUTPUT_DIR = "heatmaps"
MIN_DISTANCE_THRESHOLD = 0.2
NUM_COMPARISON_CONSTRAINTS = 5
DURATION_THRESHOLD_HOURS = 0.5
TIMEWINDOW_THRESHOLD_HOURS = 2.0
