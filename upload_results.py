#!/usr/bin/env python3
"""
upload_results.py - Upload benchmark results to experiment tracking system

This script uploads log directories and creates sub-experiments for:
- Each scenario in the study
- Each parameter range combination (if param_ranges were used)

Supports MLflow and can be extended for other experiment tracking systems.

Usage:
    python upload_results.py <study_dir> [--experiment-name NAME] [--mlflow-tracking-uri URI] [--dry-run]

Example:
    python upload_results.py Study_MLPerf_Run_2026-01-20_20-50-29 --experiment-name mlperf-benchmark
"""

import argparse
import json
import re
import shutil
import sys
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


def upload_scenario_to_mlflow(
    study_dir: Path,
    scenario_dir: Path,
    experiment_name: str,
    param_combo_name: Optional[str] = None,
    dry_run: bool = False
):
    """Upload a scenario (or param combo) to MLflow as a sub-experiment/run."""
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
        
        # List what would be uploaded
        log_files = get_log_directories(scenario_dir)
        param_combo_dir = scenario_dir / 'results' / param_combo_name if param_combo_name else None
        result_files = get_result_files(scenario_dir, param_combo_dir)
        
        print(f"  Log files ({len(log_files)}):")
        for log_file in log_files:
            print(f"    - {log_file.relative_to(study_dir)}")
        
        print(f"  Result files ({len(result_files)}):")
        for result_file in result_files:
            print(f"    - {result_file.relative_to(study_dir)}")
        
        return
    
    if not MLFLOW_AVAILABLE:
        print(f"Error: mlflow not available. Skipping {run_name}")
        return
    
    try:
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log tags
            mlflow.set_tag("scenario", scenario_name)
            if param_combo_name:
                mlflow.set_tag("param_combo", param_combo_name)
            
            # Log log files as artifacts
            log_files = get_log_directories(scenario_dir)
            if log_files:
                logs_artifact_dir = Path("logs")
                logs_artifact_dir.mkdir(exist_ok=True)
                
                for log_file in log_files:
                    artifact_path = logs_artifact_dir / log_file.name
                    shutil.copy2(log_file, artifact_path)
                
                mlflow.log_artifacts(logs_artifact_dir, artifact_path="logs")
                shutil.rmtree(logs_artifact_dir)
                print(f"  ✅ Uploaded {len(log_files)} log files")
            
            # Log result files as artifacts
            param_combo_dir = scenario_dir / 'results' / param_combo_name if param_combo_name else None
            result_files = get_result_files(scenario_dir, param_combo_dir)
            if result_files:
                results_artifact_dir = Path("results")
                results_artifact_dir.mkdir(exist_ok=True)
                
                for result_file in result_files:
                    artifact_path = results_artifact_dir / result_file.name
                    shutil.copy2(result_file, artifact_path)
                    
                    # Try to parse and log metrics from JSON
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
                
                mlflow.log_artifacts(results_artifact_dir, artifact_path="results")
                shutil.rmtree(results_artifact_dir)
                print(f"  ✅ Uploaded {len(result_files)} result files")
            
            # Log scenario directory path as parameter
            mlflow.log_param("scenario_dir", str(scenario_dir.relative_to(study_dir)))
            if param_combo_name:
                mlflow.log_param("param_combo", param_combo_name)
            
            print(f"✅ Uploaded run: {run_name}")
    
    except Exception as e:
        print(f"❌ Error uploading {run_name}: {e}")


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
    
    # Process each scenario
    for scenario_dir in scenario_dirs:
        scenario_name = parse_scenario_name(scenario_dir)
        print(f"\n📋 Processing scenario: {scenario_name}")
        
        # Detect parameter combinations
        param_combos = detect_param_combinations(scenario_dir)
        
        if len(param_combos) > 1 or (len(param_combos) == 1 and param_combos[0][0] is not None):
            # Param ranges were used - create sub-experiment for each combo
            print(f"  🔄 Found {len(param_combos)} parameter combination(s)")
            
            for param_combo_name, param_combo_dir in param_combos:
                print(f"  📦 Processing param combo: {param_combo_name}")
                upload_scenario_to_mlflow(
                    study_dir,
                    scenario_dir,
                    experiment_name,
                    param_combo_name,
                    args.dry_run
                )
        else:
            # No param ranges - single sub-experiment for scenario
            print(f"  📦 No param ranges detected")
            upload_scenario_to_mlflow(
                study_dir,
                scenario_dir,
                experiment_name,
                None,
                args.dry_run
            )
    
    print(f"\n✅ Upload complete!")


if __name__ == '__main__':
    main()
