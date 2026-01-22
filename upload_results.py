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
from typing import Dict, List, Optional, Tuple

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


def parse_vllm_server_args(log_file: Path) -> Dict[str, str]:
    """
    Parse vLLM server command line arguments from log file.
    
    Looks for lines containing 'vllm serve' command and extracts arguments.
    Returns a dictionary of argument names to values.
    """
    if not log_file or not log_file.exists():
        return {}
    
    args_dict = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Look for vllm serve command
            # Pattern: vllm serve <model> --port <port> [args...]
            # Also handle cases where it might be wrapped (e.g., "nsys profile ... vllm serve ...")
            # Match vllm serve command with arguments
            # This regex captures the command line after "vllm serve"
            # Try to match the full command line, handling both single-line and multi-line cases
            patterns = [
                # Direct vllm serve command
                r'vllm\s+serve\s+[^\s]+\s+(?:--port\s+\d+\s+)?(.*?)(?:\n|$)',
                # Command might be in a log line with other text
                r'(?:^|[\s>])(?:python[0-9]*\s+)?vllm\s+serve\s+[^\s]+\s+(?:--port\s+\d+\s+)?(.*?)(?:\n|$)',
            ]
            
            args_string = None
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if matches:
                    # Take the first match (usually the command that started the server)
                    args_string = matches[0].strip()
                    break
            
            if args_string:
                
                # Parse arguments
                # Handle both --arg value and --arg=value formats
                # Split by spaces but handle quoted values
                try:
                    parts = shlex.split(args_string)
                except ValueError:
                    # Fallback: simple split if shlex fails
                    parts = args_string.split()
                
                i = 0
                while i < len(parts):
                    arg = parts[i]
                    if arg.startswith('--'):
                        # Remove leading --
                        arg_name = arg[2:]
                        
                        # Check if it's --arg=value format
                        if '=' in arg_name:
                            arg_name, arg_value = arg_name.split('=', 1)
                            args_dict[arg_name] = arg_value
                        else:
                            # Check if next part is a value (not another --arg)
                            if i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                                args_dict[arg_name] = parts[i + 1]
                                i += 1
                            else:
                                # Boolean flag (no value)
                                args_dict[arg_name] = "true"
                    i += 1
    except Exception as e:
        print(f"  ⚠️  Warning: Could not parse server log {log_file.name}: {e}")
    
    return args_dict


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
            server_args = parse_vllm_server_args(server_log)
            if server_args:
                print(f"  Extracted vLLM args ({len(server_args)}):")
                for key, value in sorted(list(server_args.items())[:5]):  # Show first 5
                    print(f"    --{key} = {value}")
                if len(server_args) > 5:
                    print(f"    ... and {len(server_args) - 5} more")
        
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
            
            # Parse and log vLLM server arguments as tags
            server_log = get_server_log_file(scenario_dir, param_combo_name)
            if server_log:
                server_args = parse_vllm_server_args(server_log)
                if server_args:
                    for arg_name, arg_value in server_args.items():
                        # Log as tag (tags are better for filtering/searching)
                        mlflow.set_tag(f"vllm_arg_{arg_name}", str(arg_value))
                    print(f"  ✅ Extracted {len(server_args)} vLLM arguments from server log")
            
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
            
            # Also extract and log metrics from JSON result files
            param_combo_dir = scenario_dir / 'results' / param_combo_name if param_combo_name else None
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
                    
                    # Parse and log vLLM server arguments as tags (for scenario-level)
                    server_log = get_server_log_file(scenario_dir, None)
                    if server_log:
                        server_args = parse_vllm_server_args(server_log)
                        if server_args:
                            for arg_name, arg_value in server_args.items():
                                mlflow.set_tag(f"vllm_arg_{arg_name}", str(arg_value))
                            print(f"  ✅ Extracted {len(server_args)} vLLM arguments from server log")
                    
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
                server_args = parse_vllm_server_args(server_log)
                if server_args:
                    print(f"  Would extract {len(server_args)} vLLM arguments from server log")
                    for key, value in sorted(list(server_args.items())[:5]):  # Show first 5
                        print(f"    --{key} = {value}")
                    if len(server_args) > 5:
                        print(f"    ... and {len(server_args) - 5} more")
            
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
