import os
import pandas as pd
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.D4PyEventLog import D4PyEventLog
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog
from collections import defaultdict
import tempfile

def load_community_data(community_csv_path):
    """Load community assignments from CSV file"""
    df = pd.read_csv(community_csv_path)
    community_map = defaultdict(list)
    for _, row in df.iterrows():
        community_map[int(row['community'])].append(row['node'])
    return community_map

def filter_log_by_cases(full_log, case_ids):
    """Filter a log to only include specified case IDs"""
    filtered_log = EventLog()
    for trace in full_log:
        if trace.attributes['concept:name'] in case_ids:
            filtered_log.append(trace)
    return filtered_log

def prepare_logs_for_analysis(full_log, normal_cases, anomalous_cases):
    """Prepare normal and anomalous logs for analysis"""
    normal_log = filter_log_by_cases(full_log, normal_cases)
    anomalous_log = filter_log_by_cases(full_log, anomalous_cases)
    
    # Export to temp XES files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xes") as tmp_normal, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".xes") as tmp_anomalous:
        
        xes_exporter.apply(normal_log, tmp_normal.name)
        xes_exporter.apply(anomalous_log, tmp_anomalous.name)
        
        # Load Declare4Py logs
        d4py_normal = D4PyEventLog()
        d4py_normal.parse_xes_log(tmp_normal.name)

        d4py_anomalous = D4PyEventLog()
        d4py_anomalous.parse_xes_log(tmp_anomalous.name)
    
    # Clean up
    os.unlink(tmp_normal.name)
    os.unlink(tmp_anomalous.name)
    
    return d4py_normal, d4py_anomalous



def run_declare_analysis(normal_log, community_id, output_dir):
    """Run DECLARE analysis and save results"""
    # Step 1: Discovery on normal traces
    miner = DeclareMiner(
        log=normal_log,
        consider_vacuity=True,
        min_support=0.7,
        itemsets_support=0.9,
        max_declare_cardinality=1
    )
    declare_model = miner.run()

    # Step 1
    results = []
    total_traces = (normal_log.log_length)
    analyzer2 = MPDeclareAnalyzer(
            log=normal_log,
            declare_model=declare_model,
            consider_vacuity=True
        )
    normal_results = analyzer2.run()
    constraint_details=[]
    for constraint_idx, constraint in enumerate(declare_model.constraints):
        satisfied_count = sum(
            1 for trace_idx in range(total_traces)
            if normal_results.model_check_res[trace_idx][constraint_idx].state.name == "SATISFIED"
        )
        
        constraint_details.append({
            "community_id": community_id,
            "template": str(constraint['template']),
            "activities": str(constraint['activities']),
            "normal_support": satisfied_count / total_traces,
            "normal_satisfied": satisfied_count,
            "normal_total": total_traces
        })
    df_constraints= pd.DataFrame(constraint_details)
    #top_constraints = df_constraints.sort_values('support', ascending=False)
    output_path1 = os.path.join(output_dir, f"Constraints in community_{community_id}_violations.csv")
    df_constraints.to_csv(output_path1, index=False)

    return df_constraints


def analyze_communities(xes_path, community_csv_path, output_dir):
    """Main analysis function that handles all communities"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    community_map = load_community_data(community_csv_path)
    full_log = xes_importer.apply(xes_path)
    
    all_results = []
    
    # Process each community as normal
    for community_id, normal_cases in community_map.items():
        print(f"\nAnalyzing community {community_id} as normal...")
        
        # Get all other cases as anomalous
        anomalous_cases = []
        for other_id, cases in community_map.items():
            if other_id != community_id:
                anomalous_cases.extend(cases)
        
        # Prepare logs
        d4py_normal, _ = prepare_logs_for_analysis(full_log, normal_cases, anomalous_cases)
                    
        # Run analysis
        results_df = run_declare_analysis(d4py_normal, community_id, output_dir)
        
        # Add to combined results
        results_df['analysis_type'] = f'community_{community_id}_as_normal'
        all_results.append(results_df)
    
    # Save combined results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_output_path = os.path.join(output_dir, "all_community_results.csv")
    combined_results.to_csv(combined_output_path, index=False)
    print(f"\nCommunity results saved to {combined_output_path}")
    
    return combined_results
