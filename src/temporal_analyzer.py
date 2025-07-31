import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from collections import defaultdict
import ast
from config import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def clean_template(template_str: str) -> str:
    """Clean and standardize template strings"""
    return template_str.split(":")[0].replace("<Template.", "").strip()

def analyze_temporal_constraints(
    xes_path: str,
    community_csv_path: str,
    constraints_csv_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Analyze temporal aspects of constraints across communities
    """
    # Load data
    log = xes_importer.apply(xes_path)
    cluster_df = pd.read_csv(community_csv_path)
    constraints_df = pd.read_csv(constraints_csv_path)
    
    # Create trace to cluster mapping
    trace_cluster_map = dict(zip(cluster_df["node"], cluster_df["community"]))
    
    # First pass: collect global durations for each (template, A, B)
    global_durations = defaultdict(list)
    
    for _, row in constraints_df.iterrows():
        cluster_id = row["community_id"]
        template = clean_template(row["template"])
        activities = ast.literal_eval(row["activities"])
        
        if len(activities) == 1:
            A, B = activities[0], None
        elif len(activities) == 2:
            A, B = activities
        else:
            continue

        if template not in settings.DURATION_TEMPLATES:
            continue

        for trace in log:
            trace_id = trace.attributes["concept:name"]
            if trace_cluster_map.get(trace_id) != cluster_id:
                continue
                
            events = [e for e in trace if "concept:name" in e and "time:timestamp" in e]
            A_times = [e["time:timestamp"] for e in events if e["concept:name"] == A]
            B_times = [e["time:timestamp"] for e in events if B and e["concept:name"] == B]

            if A_times and B_times:
                duration = (min(B_times) - min(A_times)).total_seconds() / 3600 
                if duration >= 0:
                    global_durations[(template, A, B)].append(duration)

    # Compute global bin edges
    global_bins = {}
    for key, durations in global_durations.items():
        if len(durations) >= 2:
            min_d = min(durations)
            max_d = max(durations)
            if max_d > min_d:
                bins = np.linspace(min_d, max_d, num=101) 
                global_bins[key] = bins.tolist()

    # Second pass: calculate stats using global bins
    results = []
    
    for _, row in constraints_df.iterrows():
        cluster_id = row["community_id"]
        template = clean_template(row["template"])
        activities = ast.literal_eval(row["activities"])
        
        if len(activities) == 1:
            A, B = activities[0], None
        elif len(activities) == 2:
            A, B = activities
        else:
            continue

        duration_list = []
        timewindow_list = []

        for trace in log:
            trace_id = trace.attributes["concept:name"]
            if trace_cluster_map.get(trace_id) != cluster_id:
                continue

            events = [e for e in trace if "concept:name" in e and "time:timestamp" in e]
            A_times = [e["time:timestamp"] for e in events if e["concept:name"] == A]

            if template in settings.DURATION_TEMPLATES and B:
                B_times = [e["time:timestamp"] for e in events if e["concept:name"] == B]
                if A_times and B_times:
                    duration = (min(B_times) - min(A_times)).total_seconds() / 3600 
                    if duration >= 0:
                        duration_list.append(duration)

            if template in settings.TIMEWINDOW_TEMPLATES:
                if A_times:
                    hour = min(A_times).hour
                    timewindow_list.append(hour)

        # Use global bins if available
        histogram_bins = None
        histogram_counts = None
        if template in settings.DURATION_TEMPLATES:
            bins_key = (template, A, B)
            if bins_key in global_bins and duration_list:
                counts, bin_edges = np.histogram(duration_list, bins=global_bins[bins_key])
                histogram_bins = list(bin_edges)
                histogram_counts = list(counts)

        results.append({
            "cluster_id": cluster_id,
            "constraint_type": template,
            "A": A,
            "B": B,
            "duration_mean_mins": np.mean(duration_list) * 60 if duration_list else None,
            "duration_std_mins": np.std(duration_list) * 60 if duration_list else None,
            "timewindow_mean_hour": np.mean(timewindow_list) if timewindow_list else None,
            "timewindow_std_hour": np.std(timewindow_list) if timewindow_list else None,
            "timewindow_min": np.min(timewindow_list) if timewindow_list else None,
            "timewindow_max": np.max(timewindow_list) if timewindow_list else None,
            "count_duration_samples": len(duration_list),
            "count_timewindow_samples": len(timewindow_list),
            "duration_histogram_bins": histogram_bins,
            "duration_histogram_counts": histogram_counts
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved temporal analysis results to {output_path}")
    
    return results_df