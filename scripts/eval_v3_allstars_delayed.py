#!/usr/bin/env python3
"""
Evaluate all v3_allstars_controls_vary experiments after a 4-hour delay.
Handles the double-nested directory structure where config files are at:
experiments/v3_allstars_controls_vary/exp_name/exp_name/config.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple


def find_experiment_dirs(base_dir: str) -> List[Tuple[str, str]]:
    """
    Find all experiment directories with config.json files.
    Returns list of (outer_dir, inner_dir) tuples where inner_dir contains config.json.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return []

    exp_dirs = []

    # Look for double-nested structure: base_dir/exp_name/exp_name/config.json
    for outer_dir in sorted(base_path.iterdir()):
        if not outer_dir.is_dir():
            continue

        # Check if there's a nested directory with the same name
        inner_dir = outer_dir / outer_dir.name
        config_path = inner_dir / "config.json"

        if config_path.exists():
            exp_dirs.append((str(outer_dir), str(inner_dir)))
            print(f"Found experiment: {outer_dir.name} -> {inner_dir}")
        else:
            # Also check if config is directly in outer_dir (fallback)
            config_path_direct = outer_dir / "config.json"
            if config_path_direct.exists():
                exp_dirs.append((str(outer_dir), str(outer_dir)))
                print(f"Found experiment (direct): {outer_dir.name}")

    return exp_dirs


def wait_for_delay(hours: float):
    """Wait for specified hours with progress updates."""
    total_seconds = int(hours * 3600)
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=total_seconds)

    print(f"\n{'='*70}")
    print(f"Waiting {hours} hours before starting evaluation...")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluation will begin at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Update every 10 minutes
    update_interval = 600  # 10 minutes
    next_update = time.time() + update_interval

    while True:
        remaining = (end_time - datetime.now()).total_seconds()
        if remaining <= 0:
            break

        if time.time() >= next_update:
            hours_left = remaining / 3600
            mins_left = (remaining % 3600) / 60
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Time remaining: {int(hours_left)}h {int(mins_left)}m")
            next_update = time.time() + update_interval

        # Sleep for 1 minute between checks
        time.sleep(min(60, remaining))

    print(f"\n{'='*70}")
    print(f"Delay complete! Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def evaluate_experiment(
    exp_name: str,
    exp_dir: str,
    num_eval_trajs: int,
    random_select: bool,
    seed: int | None,
    eval_script: str,
) -> Tuple[bool, str]:
    """
    Run evaluation for a single experiment.
    Returns (success, message) tuple.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {exp_name}")
    print(f"Directory: {exp_dir}")
    print(f"{'='*70}")

    # Check if required files exist
    config_path = os.path.join(exp_dir, "config.json")
    checkpoint_path = os.path.join(exp_dir, "best.pt")

    if not os.path.exists(config_path):
        msg = f"SKIP: config.json not found at {config_path}"
        print(msg)
        return False, msg

    if not os.path.exists(checkpoint_path):
        msg = f"SKIP: best.pt not found at {checkpoint_path}"
        print(msg)
        return False, msg

    # Build command
    cmd = [
        sys.executable,
        eval_script,
        "--exp-dir", exp_dir,
        "--num-eval-trajs", str(num_eval_trajs),
    ]

    if random_select:
        cmd.append("--random")

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per evaluation
        )

        if result.returncode == 0:
            print(f"SUCCESS: {exp_name}")
            print(result.stdout)
            return True, "Success"
        else:
            msg = f"FAILED with exit code {result.returncode}"
            print(f"ERROR: {msg}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, msg

    except subprocess.TimeoutExpired:
        msg = "TIMEOUT: Evaluation took longer than 1 hour"
        print(msg)
        return False, msg

    except Exception as e:
        msg = f"EXCEPTION: {str(e)}"
        print(msg)
        return False, msg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all v3_allstars_controls_vary experiments after delay"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/common/home/st1122/Projects/mushr_mujoco_sysid/experiments/v3_allstars_controls_vary",
        help="Base directory containing all experiments",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default="/common/home/st1122/Projects/mushr_mujoco_sysid/scripts/eval.py",
        help="Path to eval.py script",
    )
    parser.add_argument(
        "--delay_hours",
        type=float,
        default=4.0,
        help="Hours to wait before starting evaluation (default: 4)",
    )
    parser.add_argument(
        "--num_eval_trajs",
        type=int,
        default=100,
        help="Number of trajectories to evaluate per experiment",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select evaluation trajectories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--no_delay",
        action="store_true",
        help="Skip the delay and start evaluation immediately",
    )

    args = parser.parse_args()

    # Find all experiments
    print(f"Searching for experiments in: {args.base_dir}")
    exp_dirs = find_experiment_dirs(args.base_dir)

    if not exp_dirs:
        print("ERROR: No experiment directories found!")
        return 1

    print(f"\nFound {len(exp_dirs)} experiments to evaluate")

    # Wait for delay
    if not args.no_delay:
        wait_for_delay(args.delay_hours)
    else:
        print("Skipping delay, starting evaluation immediately...")

    # Evaluate all experiments
    results = {}
    successful = 0
    failed = 0

    for outer_dir, inner_dir in exp_dirs:
        exp_name = os.path.basename(outer_dir)
        success, msg = evaluate_experiment(
            exp_name=exp_name,
            exp_dir=inner_dir,
            num_eval_trajs=args.num_eval_trajs,
            random_select=args.random,
            seed=args.seed,
            eval_script=args.eval_script,
        )

        results[exp_name] = {"success": success, "message": msg}
        if success:
            successful += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(exp_dirs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*70}\n")

    print("Details:")
    for exp_name, result in sorted(results.items()):
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {exp_name}: {result['message']}")

    # Save results to JSON
    results_file = os.path.join(args.base_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "args": vars(args),
                "results": results,
                "summary": {
                    "total": len(exp_dirs),
                    "successful": successful,
                    "failed": failed,
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
