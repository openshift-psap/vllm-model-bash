#!/usr/bin/env python3
"""
upload_results.py - Upload benchmark results to experiment tracking system

This script uploads log directories and creates a hierarchical structure in MLflow:
- Main experiment (parent)
  - Study sub-experiment (nested run for the study)
    - Scenario sub-experiment (nested run for each scenario)
      - Parameter combination runs (if param_ranges were used)

The script also:
- Parses vLLM server logs to extract command-line arguments
- Logs vLLM arguments as MLflow tags for easy filtering
- Uploads all log files and result files as artifacts
- Extracts metrics from JSON result files

Supports MLflow and can be extended for other experiment tracking systems.

Usage:
    python upload_results.py <study_dir> [--experiment-name NAME] [--mlflow-tracking-uri URI] [--dry-run]

Example:
    python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29 --experiment-name mlperf-benchmark
"""

import argparse
import json
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not installed. Install with: pip install mlflow")


def parse_scenario_name(scenario_dir: Path) -> str:
    """Extract scenario name from directory name (e.g., 'scenario_mlperf_offline' -> 'mlperf_offline')."""
    name = scenario_dir.name
    if name.startswith('scenario_'):
        return name[len('scenario_'):]
    return name


def detect_param_combinations(scenario_dir: Path) -> List[Tuple[str, Path]]:
    """
    Detect parameter combinations by checking for subdirectories in results/.
    
    Returns:
        List of (param_combo_name, param_combo_dir) tuples.
        If no param ranges were used, returns [(None, results_dir)].
    """
    results_dir = scenario_dir / 'results'
    if not results_dir.exists():
        return []
    
    # Check if there are subdirectories (indicating param_ranges were used)
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # Param ranges were used - each subdirectory is a param combination
        return [(d.name, d) for d in sorted(subdirs)]
    else:
        # No param ranges - single results directory
        return [(None, results_dir)]


def get_log_directories(scenario_dir: Path) -> List[Path]:
    """Get all log directories/files for a scenario."""
    logs_dir = scenario_dir / 'logs'
    if not logs_dir.exists():
        return []
    
    # Return all log files
    log_files = list(logs_dir.glob('*.log'))
    return sorted(log_files)


def get_server_log_file(scenario_dir: Path, param_combo_name: Optional[str] = None) -> Optional[Path]:
    """Get the vLLM server log file for a scenario."""
    logs_dir = scenario_dir / 'logs'
    if not logs_dir.exists():
        return None
    
    # Try to find server log file
    # Pattern: vllm_server_*.log
    server_logs = list(logs_dir.glob('vllm_server_*.log'))
    if server_logs:
        # If param_combo_name is provided, try to match it in the filename
        if param_combo_name:
            for log_file in server_logs:
                if param_combo_name in log_file.name:
                    return log_file
        # Otherwise, return the first one (or most recent if multiple)
        return sorted(server_logs)[-1] if server_logs else None
    
    return None


def parse_vllm_server_log(log_file: Path) -> Dict[str, Any]:
    """
    Parse vLLM server log file to extract:
    1. API version (from "vLLM API server version X.X.X")
    2. Non-default arguments (from "non-default args: {...}")
    
    Returns a dictionary with 'version' and 'args' keys.
    """
    if not log_file or not log_file.exists():
        return {'version': None, 'args': {}}
    
    result = {'version': None, 'args': {}}
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract API version
                # Pattern: "vLLM API server version 0.13.0"
                version_match = re.search(r'vLLM\s+API\s+server\s+version\s+([\d.]+)', line, re.IGNORECASE)
                if version_match:
                    result['version'] = version_match.group(1)
                
                # Extract non-default args
                # Pattern: "non-default args: {'key': 'value', ...}"
                if 'non-default args:' in line:
                    # Find the dictionary part after "non-default args:"
                    dict_start = line.find('non-default args:') + len('non-default args:')
                    dict_str = line[dict_start:].strip()
                    
                    # Try to parse as Python dict literal
                    try:
                        # Use eval to parse the dictionary (safe here as it's from our own logs)
                        # The dict is in the format: {'key': 'value', ...}
                        args_dict = eval(dict_str)
                        if isinstance(args_dict, dict):
                            result['args'] = {str(k): str(v) for k, v in args_dict.items()}
                    except Exception:
                        # Fallback: try to extract key-value pairs manually
                        # Look for patterns like 'key': 'value' or 'key': value
                        kv_pattern = r"'([^']+)':\s*(?:'([^']*)'|([^,}]+))"
                        matches = re.findall(kv_pattern, dict_str)
                        for match in matches:
                            key = match[0]
                            value = match[1] if match[1] else match[2].strip()
                            result['args'][key] = value
                    
                    # Only process the first occurrence
                    break
    except Exception as e:
        print(f"  ⚠️  Warning: Could not parse server log {log_file.name}: {e}")
    
    return result


def parse_vllm_server_args(log_file: Path) -> Dict[str, str]:
    """
    Parse vLLM server log file and return non-default arguments.
    This is a convenience wrapper around parse_vllm_server_log.
    """
    log_data = parse_vllm_server_log(log_file)
    return log_data.get('args', {})


def get_result_files(scenario_dir: Path, param_combo_dir: Optional[Path] = None) -> List[Path]:
    """Get all result files for a scenario (optionally filtered by param combo)."""
    if param_combo_dir:
        search_dir = param_combo_dir
    else:
        search_dir = scenario_dir / 'results'
    
    if not search_dir.exists():
        return []
    
    # Find all JSON result files
    result_files = list(search_dir.glob('*.json'))
    return sorted(result_files)


def find_mlperf_summary_file(scenario_dir: Path, param_combo_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find mlperf_log_summary.txt file in results directory.
    
    When param_ranges are used, the structure is:
    - scenario_dir/results/param_combo_name/mlperf_log_summary.txt
    
    When no param_ranges, the structure is:
    - scenario_dir/results/mlperf_log_summary.txt
    - or scenario_dir/results/subdir/mlperf_log_summary.txt (if MLPerf creates subdirs)
    """
    if param_combo_dir:
        # When param_combo_dir is provided, it's already the param combo subdirectory
        # e.g., scenario_dir/results/param_combo_name
        search_dir = param_combo_dir
    else:
        # No param ranges - search in results directory
        search_dir = scenario_dir / 'results'
    
    if not search_dir.exists():
        return None
    
    # First, check directly in the search directory
    summary_file = search_dir / 'mlperf_log_summary.txt'
    if summary_file.exists():
        return summary_file
    
    # If not found directly, search recursively in subdirectories
    # This handles cases where MLPerf creates additional subdirectories
    for summary_file in search_dir.rglob('mlperf_log_summary.txt'):
        if summary_file.is_file():
            return summary_file
    
    return None


def parse_mlperf_summary(summary_file: Path) -> Dict[str, Any]:
    """
    Parse MLPerf summary file and extract metrics.
    
    Returns a dictionary of metric names to values.
    """
    metrics = {}
    
    if not summary_file or not summary_file.exists():
        return metrics
    
    try:
        with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Extract key metrics
            # Samples per second
            samples_match = re.search(r'Samples per second:\s*([\d.]+)', content)
            if samples_match:
                metrics['samples_per_second'] = float(samples_match.group(1))
            
            # Tokens per second
            tokens_match = re.search(r'Tokens per second:\s*([\d.]+)', content)
            if tokens_match:
                metrics['tokens_per_second'] = float(tokens_match.group(1))
            
            # Latency metrics (convert from nanoseconds to milliseconds)
            latency_patterns = {
                'min_latency_ms': r'Min latency \(ns\)\s*:\s*(\d+)',
                'max_latency_ms': r'Max latency \(ns\)\s*:\s*(\d+)',
                'mean_latency_ms': r'Mean latency \(ns\)\s*:\s*(\d+)',
                'p50_latency_ms': r'50\.00 percentile latency \(ns\)\s*:\s*(\d+)',
                'p90_latency_ms': r'90\.00 percentile latency \(ns\)\s*:\s*(\d+)',
                'p95_latency_ms': r'95\.00 percentile latency \(ns\)\s*:\s*(\d+)',
                'p97_latency_ms': r'97\.00 percentile latency \(ns\)\s*:\s*(\d+)',
                'p99_latency_ms': r'99\.00 percentile latency \(ns\)\s*:\s*(\d+)',
                'p99_9_latency_ms': r'99\.90 percentile latency \(ns\)\s*:\s*(\d+)',
            }
            
            for metric_name, pattern in latency_patterns.items():
                match = re.search(pattern, content)
                if match:
                    # Convert nanoseconds to milliseconds
                    ns_value = int(match.group(1))
                    metrics[metric_name] = ns_value / 1_000_000.0
            
            # Test parameters
            param_patterns = {
                'samples_per_query': r'samples_per_query\s*:\s*(\d+)',
                'target_qps': r'target_qps\s*:\s*([\d.]+)',
                'ttft_latency_ns': r'ttft_latency \(ns\):\s*(\d+)',
                'tpot_latency_ns': r'tpot_latency \(ns\):\s*(\d+)',
                'max_async_queries': r'max_async_queries\s*:\s*(\d+)',
                'min_duration_ms': r'min_duration \(ms\):\s*(\d+)',
                'max_duration_ms': r'max_duration \(ms\):\s*(\d+)',
                'min_query_count': r'min_query_count\s*:\s*(\d+)',
                'max_query_count': r'max_query_count\s*:\s*(\d+)',
            }
            
            for param_name, pattern in param_patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    # Try to convert to appropriate type
                    try:
                        if '.' in value:
                            metrics[param_name] = float(value)
                        else:
                            metrics[param_name] = int(value)
                    except ValueError:
                        metrics[param_name] = value
            
            # Extract scenario and mode
            scenario_match = re.search(r'Scenario\s*:\s*(\w+)', content)
            if scenario_match:
                metrics['mlperf_scenario'] = scenario_match.group(1)
            
            mode_match = re.search(r'Mode\s*:\s*(\w+)', content)
            if mode_match:
                metrics['mlperf_mode'] = mode_match.group(1)
            
            # Extract result validity
            result_match = re.search(r'Result is\s*:\s*(\w+)', content)
            if result_match:
                metrics['mlperf_result'] = result_match.group(1)
    
    except Exception as e:
        print(f"  ⚠️  Warning: Could not parse MLPerf summary {summary_file.name}: {e}")
    
    return metrics


def load_config(study_dir: Path) -> Optional[Dict]:
    """Load the config.yaml from the study directory."""
    config_file = study_dir / 'config.yaml'
    if not config_file.exists():
        return None
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return None


def create_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    dry_run: bool = False
) -> Optional[str]:
    """Create or get MLflow experiment."""
    if not MLFLOW_AVAILABLE:
        print("Error: mlflow not available. Install with: pip install mlflow")
        return None
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    if dry_run:
        print(f"[DRY RUN] Would create/get experiment: {experiment_name}")
        return experiment_name
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_name
    except Exception as e:
        print(f"Error creating MLflow experiment: {e}")
        return None


def upload_directory_structure(source_dir: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Count files in each subdirectory of a scenario directory.
    
    Returns a dictionary mapping subdirectory names to file counts.
    """
    counts = {}
    if not source_dir.exists():
        return counts
    
    for item in source_dir.iterdir():
        if item.is_dir():
            # Count files recursively
            file_count = sum(1 for _ in item.rglob('*') if _.is_file())
            counts[item.name] = file_count
        elif item.is_file():
            # Files directly in the root
            if 'root' not in counts:
                counts['root'] = 0
            counts['root'] += 1
    
    return counts


def upload_scenario_to_mlflow(
    study_dir: Path,
    scenario_dir: Path,
    parent_run_id: Optional[str],
    param_combo_name: Optional[str] = None,
    dry_run: bool = False
):
    """Upload a scenario (or param combo) to MLflow as a nested run."""
    scenario_name = parse_scenario_name(scenario_dir)
    
    # Create run name
    if param_combo_name:
        run_name = f"{scenario_name}_{param_combo_name}"
    else:
        run_name = scenario_name
    
    if dry_run:
        print(f"\n[DRY RUN] Would create run: {run_name}")
        print(f"  Scenario: {scenario_name}")
        if param_combo_name:
            print(f"  Param combo: {param_combo_name}")
        if parent_run_id:
            print(f"  Parent run ID: {parent_run_id}")
        
        # List directory structure
        dir_counts = upload_directory_structure(scenario_dir, dry_run=True)
        print(f"  Directory structure:")
        for dir_name, file_count in sorted(dir_counts.items()):
            print(f"    {dir_name}/ ({file_count} files)")
        
        server_log = get_server_log_file(scenario_dir, param_combo_name)
        if server_log:
            print(f"  Server log: {server_log.relative_to(study_dir)}")
            log_data = parse_vllm_server_log(server_log)
            if log_data.get('version'):
                print(f"  vLLM version: {log_data['version']}")
            if log_data.get('args'):
                print(f"  Extracted vLLM args ({len(log_data['args'])}):")
                for key, value in sorted(list(log_data['args'].items())[:5]):  # Show first 5
                    print(f"    {key} = {value}")
                if len(log_data['args']) > 5:
                    print(f"    ... and {len(log_data['args']) - 5} more")
        
        # Check for MLPerf summary
        param_combo_dir = scenario_dir / 'results' / param_combo_name if param_combo_name else None
        mlperf_summary = find_mlperf_summary_file(scenario_dir, param_combo_dir)
        if mlperf_summary:
            print(f"  MLPerf summary: {mlperf_summary.relative_to(study_dir)}")
            mlperf_metrics = parse_mlperf_summary(mlperf_summary)
            if mlperf_metrics:
                print(f"  Extracted MLPerf metrics ({len(mlperf_metrics)}):")
                for key, value in sorted(list(mlperf_metrics.items())[:5]):  # Show first 5
                    print(f"    {key} = {value}")
                if len(mlperf_metrics) > 5:
                    print(f"    ... and {len(mlperf_metrics) - 5} more")
        
        return None
    
    if not MLFLOW_AVAILABLE:
        print(f"Error: mlflow not available. Skipping {run_name}")
        return None
    
    try:
        # Start nested run under parent
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            # Log tags
            mlflow.set_tag("scenario", scenario_name)
            if param_combo_name:
                mlflow.set_tag("param_combo", param_combo_name)
            
            # Parse and log vLLM server log data
            server_log = get_server_log_file(scenario_dir, param_combo_name)
            if server_log:
                log_data = parse_vllm_server_log(server_log)
                
                # Log API version as tag
                if log_data.get('version'):
                    mlflow.set_tag("vllm_version", log_data['version'])
                
                # Log non-default arguments as tags
                if log_data.get('args'):
                    for arg_name, arg_value in log_data['args'].items():
                        # Log as tag (tags are better for filtering/searching)
                        mlflow.set_tag(f"vllm_arg_{arg_name}", str(arg_value))
                    print(f"  ✅ Extracted vLLM version {log_data.get('version', 'unknown')} and {len(log_data['args'])} arguments from server log")
            
            # Upload entire scenario directory structure preserving folder hierarchy
            # This will upload logs/, results/, profiles/, and all their contents
            if scenario_dir.exists():
                # Create a temporary copy to preserve structure
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    # Copy entire scenario directory structure
                    scenario_artifact_dir = temp_path / scenario_dir.name
                    shutil.copytree(scenario_dir, scenario_artifact_dir, dirs_exist_ok=True)
                    
                    # Upload the entire directory structure
                    mlflow.log_artifacts(scenario_artifact_dir, artifact_path="scenario")
                    
                    # Count files uploaded
                    dir_counts = upload_directory_structure(scenario_dir)
                    total_files = sum(dir_counts.values())
                    print(f"  ✅ Uploaded scenario directory structure ({total_files} files)")
                    for dir_name, file_count in sorted(dir_counts.items()):
                        if file_count > 0:
                            print(f"    - {dir_name}/ ({file_count} files)")
            
            # Parse and log MLPerf summary metrics
            param_combo_dir = scenario_dir / 'results' / param_combo_name if param_combo_name else None
            mlperf_summary = find_mlperf_summary_file(scenario_dir, param_combo_dir)
            if mlperf_summary:
                mlperf_metrics = parse_mlperf_summary(mlperf_summary)
                if mlperf_metrics:
                    for metric_name, metric_value in mlperf_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)
                        else:
                            # Log non-numeric values as parameters
                            mlflow.log_param(metric_name, str(metric_value))
                    print(f"  ✅ Extracted and logged {len(mlperf_metrics)} MLPerf metrics")
            
            # Also extract and log metrics from JSON result files
            result_files = get_result_files(scenario_dir, param_combo_dir)
            if result_files:
                for result_file in result_files:
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                # Log numeric metrics
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        mlflow.log_metric(key, value)
                    except Exception:
                        pass  # Skip if not JSON or not parseable
            
            # Log scenario directory path as parameter
            mlflow.log_param("scenario_dir", str(scenario_dir.relative_to(study_dir)))
            if param_combo_name:
                mlflow.log_param("param_combo", param_combo_name)
            
            print(f"✅ Uploaded run: {run_name}")
            return run.info.run_id
    
    except Exception as e:
        print(f"❌ Error uploading {run_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Upload benchmark results to experiment tracking system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to MLflow with default experiment name
  python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29

  # Upload with custom experiment name
  python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29 --experiment-name mlperf-benchmark

  # Dry run to see what would be uploaded
  python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29 --dry-run

  # Use custom MLflow tracking URI
  python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29 --mlflow-tracking-uri http://localhost:5000
        """
    )
    
    parser.add_argument(
        'study_dir',
        type=str,
        help='Path to study directory (e.g., Study_MLPerf_Run_2026-01-20_20-50-29)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (default: derived from study directory name)'
    )
    
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: use MLFLOW_TRACKING_URI env var or local file store)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - show what would be uploaded without actually uploading'
    )
    
    args = parser.parse_args()
    
    # Validate study directory
    study_dir = Path(args.study_dir).resolve()
    if not study_dir.exists():
        print(f"Error: Study directory does not exist: {study_dir}")
        sys.exit(1)
    
    if not study_dir.is_dir():
        print(f"Error: Not a directory: {study_dir}")
        sys.exit(1)
    
    # Find all scenario directories
    scenario_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir() and d.name.startswith('scenario_')])
    
    if not scenario_dirs:
        print(f"Warning: No scenario directories found in {study_dir}")
        print("Expected directories named 'scenario_*'")
        sys.exit(1)
    
    print(f"📁 Study directory: {study_dir}")
    print(f"📊 Found {len(scenario_dirs)} scenario(s)")
    
    # Load config to get study name
    config = load_config(study_dir)
    
    # Determine experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    elif config and 'defaults' in config and 'study_dir' in config['defaults']:
        # Use study_dir from config (without timestamp)
        experiment_name = config['defaults']['study_dir']
    else:
        # Derive from directory name (remove timestamp)
        study_name = study_dir.name
        # Remove timestamp pattern (e.g., _2026-01-20_20-50-29)
        experiment_name = re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', '', study_name)
    
    print(f"🔬 Experiment name: {experiment_name}")
    
    # Create/get MLflow experiment
    if not args.dry_run:
        exp_name = create_mlflow_experiment(experiment_name, args.mlflow_tracking_uri, args.dry_run)
        if not exp_name:
            print("Error: Could not create/get MLflow experiment")
            sys.exit(1)
    
    # Create study sub-experiment (nested under main experiment)
    study_name = study_dir.name
    if not args.dry_run:
        print(f"\n📚 Creating study sub-experiment: {study_name}")
        with mlflow.start_run(run_name=study_name, nested=True) as study_run:
            # Log study metadata
            mlflow.set_tag("study_name", study_name)
            mlflow.log_param("study_dir", str(study_dir))
            
            # Load and log config if available
            if config:
                try:
                    import yaml
                    config_str = yaml.dump(config, default_flow_style=False)
                    mlflow.log_text(config_str, artifact_file="config.yaml")
                except Exception:
                    pass
            
            study_run_id = study_run.info.run_id
            
            # Process each scenario (nested under study)
            for scenario_dir in scenario_dirs:
                scenario_name = parse_scenario_name(scenario_dir)
                print(f"\n📋 Processing scenario: {scenario_name}")
                
                # Create scenario sub-experiment (nested under study)
                with mlflow.start_run(run_name=scenario_name, nested=True) as scenario_run:
                    scenario_run_id = scenario_run.info.run_id
                    
                    # Log scenario metadata
                    mlflow.set_tag("scenario", scenario_name)
                    mlflow.log_param("scenario_dir", str(scenario_dir.relative_to(study_dir)))
                    
                    # Parse and log vLLM server log data (for scenario-level)
                    server_log = get_server_log_file(scenario_dir, None)
                    if server_log:
                        log_data = parse_vllm_server_log(server_log)
                        
                        # Log API version as tag
                        if log_data.get('version'):
                            mlflow.set_tag("vllm_version", log_data['version'])
                        
                        # Log non-default arguments as tags
                        if log_data.get('args'):
                            for arg_name, arg_value in log_data['args'].items():
                                mlflow.set_tag(f"vllm_arg_{arg_name}", str(arg_value))
                            print(f"  ✅ Extracted vLLM version {log_data.get('version', 'unknown')} and {len(log_data['args'])} arguments from server log")
                    
                    # Detect parameter combinations
                    param_combos = detect_param_combinations(scenario_dir)
                    
                    if len(param_combos) > 1 or (len(param_combos) == 1 and param_combos[0][0] is not None):
                        # Param ranges were used - create runs for each combo
                        print(f"  🔄 Found {len(param_combos)} parameter combination(s)")
                        
                        for param_combo_name, param_combo_dir in param_combos:
                            print(f"  📦 Processing param combo: {param_combo_name}")
                            upload_scenario_to_mlflow(
                                study_dir,
                                scenario_dir,
                                scenario_run_id,
                                param_combo_name,
                                args.dry_run
                            )
                    else:
                        # No param ranges - upload artifacts directly to scenario run
                        print(f"  📦 No param ranges detected - uploading to scenario run")
                        
                        # Upload entire scenario directory structure preserving folder hierarchy
                        if scenario_dir.exists():
                            with tempfile.TemporaryDirectory() as temp_dir:
                                temp_path = Path(temp_dir)
                                # Copy entire scenario directory structure
                                scenario_artifact_dir = temp_path / scenario_dir.name
                                shutil.copytree(scenario_dir, scenario_artifact_dir, dirs_exist_ok=True)
                                
                                # Upload the entire directory structure
                                mlflow.log_artifacts(scenario_artifact_dir, artifact_path="scenario")
                                
                                # Count files uploaded
                                dir_counts = upload_directory_structure(scenario_dir)
                                total_files = sum(dir_counts.values())
                                print(f"  ✅ Uploaded scenario directory structure ({total_files} files)")
                                for dir_name, file_count in sorted(dir_counts.items()):
                                    if file_count > 0:
                                        print(f"    - {dir_name}/ ({file_count} files)")
                        
                        # Parse and log MLPerf summary metrics
                        mlperf_summary = find_mlperf_summary_file(scenario_dir, None)
                        if mlperf_summary:
                            mlperf_metrics = parse_mlperf_summary(mlperf_summary)
                            if mlperf_metrics:
                                for metric_name, metric_value in mlperf_metrics.items():
                                    if isinstance(metric_value, (int, float)):
                                        mlflow.log_metric(metric_name, metric_value)
                                    else:
                                        # Log non-numeric values as parameters
                                        mlflow.log_param(metric_name, str(metric_value))
                                print(f"  ✅ Extracted and logged {len(mlperf_metrics)} MLPerf metrics")
                        
                        # Also extract and log metrics from JSON result files
                        result_files = get_result_files(scenario_dir, None)
                        if result_files:
                            for result_file in result_files:
                                try:
                                    with open(result_file, 'r') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            # Log numeric metrics
                                            for key, value in data.items():
                                                if isinstance(value, (int, float)):
                                                    mlflow.log_metric(key, value)
                                except Exception:
                                    pass  # Skip if not JSON or not parseable
    else:
        # Dry run mode
        print(f"\n[DRY RUN] Would create study sub-experiment: {study_name}")
        
        # Process each scenario
        for scenario_dir in scenario_dirs:
            scenario_name = parse_scenario_name(scenario_dir)
            print(f"\n📋 Processing scenario: {scenario_name}")
            print(f"  [DRY RUN] Would create scenario sub-experiment: {scenario_name}")
            
            # Parse and log vLLM server arguments (dry run)
            server_log = get_server_log_file(scenario_dir, None)
            if server_log:
                log_data = parse_vllm_server_log(server_log)
                if log_data.get('version'):
                    print(f"  Would extract vLLM version: {log_data['version']}")
                if log_data.get('args'):
                    print(f"  Would extract {len(log_data['args'])} vLLM arguments from server log")
                    for key, value in sorted(list(log_data['args'].items())[:5]):  # Show first 5
                        print(f"    {key} = {value}")
                    if len(log_data['args']) > 5:
                        print(f"    ... and {len(log_data['args']) - 5} more")
            
            # Check for MLPerf summary (dry run)
            mlperf_summary = find_mlperf_summary_file(scenario_dir, None)
            if mlperf_summary:
                print(f"  Would parse MLPerf summary: {mlperf_summary.relative_to(study_dir)}")
                mlperf_metrics = parse_mlperf_summary(mlperf_summary)
                if mlperf_metrics:
                    print(f"  Would extract {len(mlperf_metrics)} MLPerf metrics")
                    for key, value in sorted(list(mlperf_metrics.items())[:5]):  # Show first 5
                        print(f"    {key} = {value}")
                    if len(mlperf_metrics) > 5:
                        print(f"    ... and {len(mlperf_metrics) - 5} more")
            
            # Detect parameter combinations
            param_combos = detect_param_combinations(scenario_dir)
            
            if len(param_combos) > 1 or (len(param_combos) == 1 and param_combos[0][0] is not None):
                # Param ranges were used - create runs for each combo
                print(f"  🔄 Found {len(param_combos)} parameter combination(s)")
                
                for param_combo_name, param_combo_dir in param_combos:
                    print(f"  📦 Processing param combo: {param_combo_name}")
                    upload_scenario_to_mlflow(
                        study_dir,
                        scenario_dir,
                        None,
                        param_combo_name,
                        args.dry_run
                    )
            else:
                # No param ranges - would upload directly to scenario run
                print(f"  📦 No param ranges detected - would upload to scenario run")
                dir_counts = upload_directory_structure(scenario_dir, dry_run=True)
                total_files = sum(dir_counts.values())
                print(f"    Would upload scenario directory structure ({total_files} files):")
                for dir_name, file_count in sorted(dir_counts.items()):
                    if file_count > 0:
                        print(f"      - {dir_name}/ ({file_count} files)")
    
    print(f"\n✅ Upload complete!")


if __name__ == '__main__':
    main()
