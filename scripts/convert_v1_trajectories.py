#!/usr/bin/env python3
"""
Convert sysid_trajs (v1) data into stepwise trajectories.

The original v1 data consists of:
- Plan files: single constant steering_angle and velocity_desired for entire trajectory
- Traj vels files: timestamped state trajectories with dt, x, y, theta, xDot, yDot, thetaDot

This script converts them to stepwise format:
x y theta xDot yDot thetaDot steering_angle velocity_desired
where each line corresponds to a single timestep with the constant control applied throughout.
"""

import os
import numpy as np
from pathlib import Path


def read_plan_file(filepath):
    """
    Read a plan file and return the single constant control.

    Returns:
        Tuple of (steering_angle, velocity_desired)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 2:
            steering = float(parts[0])
            velocity = float(parts[1])
            return steering, velocity

    raise ValueError(f"No valid control line found in {filepath}")


def read_traj_vels_file(filepath):
    """
    Read a trajectory velocities file and return state data.

    Returns:
        List of tuples: (x, y, theta, xDot, yDot, thetaDot)
    """
    states = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 7:
            # Format: dt x y theta xDot yDot thetaDot [optional 8th column]
            x = float(parts[1])
            y = float(parts[2])
            theta = float(parts[3])
            xDot = float(parts[4])
            yDot = float(parts[5])
            thetaDot = float(parts[6])

            states.append((x, y, theta, xDot, yDot, thetaDot))

    return states


def convert_trajectory(plan_file, traj_vels_file, output_file):
    """Convert a single trajectory from v1 format to stepwise format."""
    # Read input files
    steering, velocity = read_plan_file(plan_file)
    states = read_traj_vels_file(traj_vels_file)

    # Write output file
    with open(output_file, 'w') as f:
        f.write("# x y theta xDot yDot thetaDot steering_angle velocity_desired\n")
        for x, y, theta, xDot, yDot, thetaDot in states:
            f.write(f"{x} {y} {theta} {xDot} {yDot} {thetaDot} {steering} {velocity}\n")

    print(f"Converted: {output_file.name} ({len(states)} timesteps)")


def main():
    # Setup paths
    input_dir = Path("data/sysid_trajs")
    output_dir = Path("data/sysid_trajs_v1")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all plan files
    plan_files = sorted(input_dir.glob("plan_*.txt"))

    print(f"Found {len(plan_files)} plan files to convert")

    converted_count = 0
    for plan_file in plan_files:
        # Extract ID from filename (e.g., plan_000.txt -> 000)
        file_id = plan_file.stem.replace("plan_", "")

        # Find corresponding traj_vels file
        traj_vels_file = input_dir / f"traj_vels_{file_id}.txt"

        if not traj_vels_file.exists():
            print(f"Warning: No matching traj_vels file for {plan_file.name}, skipping")
            continue

        # Create output filename
        output_file = output_dir / f"traj_{file_id}.txt"

        # Convert
        try:
            convert_trajectory(plan_file, traj_vels_file, output_file)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {plan_file.name}: {e}")

    print(f"\nConversion complete: {converted_count}/{len(plan_files)} files converted successfully")


if __name__ == "__main__":
    main()
