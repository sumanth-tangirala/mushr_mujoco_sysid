import os
import re
import numpy as np
from typing import List, Optional, Tuple

L = 0.31
MIN_PHYSICAL_RADIUS = L / 2  # Minimum radius for Ackermann geometry

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "sysid_trajs")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "sysid_dataset.txt")
FILTERED_OUTPUT = os.path.join(BASE_DIR, "data", "sysid_dataset_filtered.txt")
VELOCITY_ONLY_OUTPUT = os.path.join(BASE_DIR, "data", "sysid_dataset_velocity_only.txt")
SHUFFLED_INDICES_FILE = os.path.join(BASE_DIR, "data", "shuffled_indices.txt")


def _read_first_non_comment_line(path: str) -> Optional[str]:
    with open(path, "r") as file:
        for raw in file:
            if raw.lstrip().startswith("#"):
                continue
            return raw.strip()
    return None


def _parse_circle_file(circle_path: str) -> Tuple[float, float]:
    line = _read_first_non_comment_line(circle_path)
    if line is None:
        raise ValueError(f"No data line in {circle_path}")
    parts = line.split()
    radius = float(parts[0])
    error = float(parts[3])
    return radius, error


def _is_radius_valid(radius: float) -> bool:
    """Check if radius is physically valid for Ackermann geometry."""
    return radius >= MIN_PHYSICAL_RADIUS


def _compute_steering_from_radius(radius: float) -> float:
    """Compute steering angle from radius. Returns NaN for invalid radii."""
    if not _is_radius_valid(radius):
        return np.nan
    beta = np.arcsin(L / (2 * radius))
    steering_real = np.arctan(2 * np.tan(beta))
    return steering_real


def _read_plan(plan_path: str) -> Tuple[float, float]:
    line = _read_first_non_comment_line(plan_path)
    if line is None:
        raise ValueError(f"No data line in {plan_path}")
    parts = line.split()
    steering_u = float(parts[0])
    velocity_u = float(parts[1])
    return steering_u, velocity_u


def _read_traj_vels(traj_vels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    dts: List[float] = []
    with open(traj_vels_path, "r") as file:
        for raw in file:
            if raw.lstrip().startswith("#"):
                continue
            parts = raw.split()
            dts.append(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            theta = float(parts[3])
            x_dot = float(parts[4])
            y_dot = float(parts[5])
            theta_dot = float(parts[6])
            rows.append([x, y, theta, x_dot, y_dot, theta_dot])
    return np.asarray(rows, dtype=float), np.asarray(dts, dtype=float)


def load_traj_data(id: int, data_dir: str = DATA_DIR):
    traj_vels_file_path = os.path.join(data_dir, f"traj_vels_{id}.txt")
    plan_file_path = os.path.join(data_dir, f"plan_{id}.txt")
    circle_file_path = os.path.join(data_dir, f"cirlce_{id}.txt")
    if not os.path.exists(circle_file_path):
        circle_file_path = os.path.join(data_dir, f"circle_{id}.txt")
        if not os.path.exists(circle_file_path):
            return None, None, None, None

    try:
        radius, error = _parse_circle_file(circle_file_path)
    except Exception:
        return None, None, None, None
    if error > 1.0:
        return None, None, None, None

    steering_real = _compute_steering_from_radius(radius)
    try:
        traj_vels, dts = _read_traj_vels(traj_vels_file_path)
        steering_u, velocity_u = _read_plan(plan_file_path)
    except Exception:
        return None, None, None, None
    return (
        traj_vels,
        np.asarray([steering_u, velocity_u], dtype=float),
        float(steering_real),
        dts,
    )


def _extract_ids(paths: List[str], pattern: str) -> set:
    regex = re.compile(pattern)
    ids = set()
    for p in paths:
        m = regex.search(os.path.basename(p))
        if m:
            try:
                ids.add(int(m.group(1)))
            except Exception:
                continue
    return ids


def find_matching_ids(data_dir: str = DATA_DIR, verbose: bool = False) -> List[int]:
    files = os.listdir(data_dir)
    traj_ids = _extract_ids(files, r"traj_vels_(\d+)\.txt$")
    plan_ids = _extract_ids(files, r"plan_(\d+)\.txt$")
    circle_ids = _extract_ids(files, r"(?:cirlce|circle)_(\d+)\.txt$")

    if verbose:
        print(f"  Found {len(traj_ids)} traj_vels files")
        print(f"  Found {len(plan_ids)} plan files")
        print(f"  Found {len(circle_ids)} circle/cirlce files")
        if traj_ids and plan_ids and circle_ids:
            print(f"  Sample traj_vels IDs: {sorted(list(traj_ids))[:5]}")
            print(f"  Sample plan IDs: {sorted(list(plan_ids))[:5]}")
            print(f"  Sample circle IDs: {sorted(list(circle_ids))[:5]}")

    common = sorted(traj_ids & plan_ids & circle_ids)
    return common


def load_shuffled_indices(
    shuffled_indices_path: str = SHUFFLED_INDICES_FILE,
) -> List[int]:
    """
    Load trajectory IDs from the shuffled indices file.

    Args:
        shuffled_indices_path: Path to the shuffled indices file

    Returns:
        List of trajectory IDs in shuffled order
    """
    ids = []
    with open(shuffled_indices_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(int(line))
    return ids


def split_train_eval_ids(
    num_eval_trajectories: int, shuffled_indices_path: str = SHUFFLED_INDICES_FILE
) -> Tuple[List[int], List[int]]:
    """
    Split shuffled trajectory IDs into training and evaluation sets.

    Args:
        num_eval_trajectories: Number of trajectories to reserve for evaluation (taken from the end of the list)
        shuffled_indices_path: Path to the shuffled indices file

    Returns:
        Tuple of (train_ids, eval_ids)
    """
    all_ids = load_shuffled_indices(shuffled_indices_path)

    if num_eval_trajectories > len(all_ids):
        raise ValueError(
            f"num_eval_trajectories ({num_eval_trajectories}) cannot be greater than total IDs ({len(all_ids)})"
        )

    # Last num_eval_trajectories for evaluation, rest for training
    eval_ids = all_ids[-num_eval_trajectories:] if num_eval_trajectories > 0 else []
    train_ids = (
        all_ids[:-num_eval_trajectories] if num_eval_trajectories > 0 else all_ids
    )

    return train_ids, eval_ids


def build_dataset_from_ids(
    ids: List[int],
    data_dir: str = DATA_DIR,
    filter_invalid_radius: bool = False,
    velocity_only: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build dataset from a specific list of trajectory IDs.

    Args:
        ids: List of trajectory IDs to include in the dataset
        data_dir: Directory containing trajectory files
        filter_invalid_radius: If True, skip trajectories with physically invalid radii
        velocity_only: If True, only predict velocity states (no steering)
        verbose: If True, print progress information

    Returns:
        Tuple of (X, Y, used_ids) where X is inputs, Y is outputs, and used_ids are the IDs successfully loaded
    """
    total = len(ids)
    if verbose:
        print(f"Building dataset from {total} trajectory IDs...")

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    used_ids: List[int] = []
    skipped = 0
    skipped_invalid_radius = 0

    for idx, id_ in enumerate(ids, 1):
        if verbose and (idx % 10 == 0 or idx == total):
            print(f"Processing {idx}/{total} (ID: {id_})...", end="\r")

        traj_vels, plan, steering_real, _ = load_traj_data(id_, data_dir=data_dir)
        if traj_vels is None or plan is None:
            skipped += 1
            continue
        if traj_vels.shape[0] < 2:
            skipped += 1
            continue

        # Filter invalid radius if requested
        if filter_invalid_radius and (steering_real is None or np.isnan(steering_real)):
            skipped_invalid_radius += 1
            continue

        xdot1 = traj_vels[:-1, 3]
        ydot1 = traj_vels[:-1, 4]
        thetadot1 = traj_vels[:-1, 5]
        steering_u = np.full_like(xdot1, plan[0], dtype=float)
        velocity_u = np.full_like(xdot1, plan[1], dtype=float)
        X = np.stack([xdot1, ydot1, thetadot1, steering_u, velocity_u], axis=1)

        if velocity_only:
            # Only predict next velocity states (no steering)
            xdot2 = traj_vels[1:, 3]
            ydot2 = traj_vels[1:, 4]
            thetadot2 = traj_vels[1:, 5]
            Y = np.stack([xdot2, ydot2, thetadot2], axis=1)
        else:
            # Predict steering and next velocity states
            steering_real_vec = np.full_like(xdot1, steering_real, dtype=float)
            xdot2 = traj_vels[1:, 3]
            ydot2 = traj_vels[1:, 4]
            thetadot2 = traj_vels[1:, 5]
            Y = np.stack([steering_real_vec, xdot2, ydot2, thetadot2], axis=1)

        X_chunks.append(X)
        Y_chunks.append(Y)
        used_ids.append(id_)

    if verbose:
        print()
        print(f"Successfully loaded: {len(used_ids)}/{total}")
        print(f"Skipped (error or short traj): {skipped}/{total}")
        if filter_invalid_radius:
            print(
                f"Skipped (invalid radius < {MIN_PHYSICAL_RADIUS:.3f}m): {skipped_invalid_radius}/{total}"
            )

    if not X_chunks:
        output_dim = 3 if velocity_only else 4
        return np.empty((0, 5), dtype=float), np.empty((0, output_dim), dtype=float), []
    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    return X_all, Y_all, used_ids


def build_dataset(
    data_dir: str = DATA_DIR, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if verbose:
        print("Scanning directory for matching file sets...")
    ids = find_matching_ids(data_dir, verbose=verbose)
    total = len(ids)
    if verbose:
        print(f"Found {total} matching file sets (traj_vels + plan + circle)")

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    used_ids: List[int] = []
    skipped = 0

    for idx, id_ in enumerate(ids, 1):
        if verbose and (idx % 10 == 0 or idx == total):
            print(f"Processing {idx}/{total} (ID: {id_})...", end="\r")

        traj_vels, plan, steering_real, _ = load_traj_data(id_, data_dir=data_dir)
        if traj_vels is None or plan is None or steering_real is None:
            skipped += 1
            continue
        if traj_vels.shape[0] < 2:
            skipped += 1
            continue
        xdot1 = traj_vels[:-1, 3]
        ydot1 = traj_vels[:-1, 4]
        thetadot1 = traj_vels[:-1, 5]
        steering_u = np.full_like(xdot1, plan[0], dtype=float)
        velocity_u = np.full_like(xdot1, plan[1], dtype=float)
        X = np.stack([xdot1, ydot1, thetadot1, steering_u, velocity_u], axis=1)
        steering_real_vec = np.full_like(xdot1, steering_real, dtype=float)
        xdot2 = traj_vels[1:, 3]
        ydot2 = traj_vels[1:, 4]
        thetadot2 = traj_vels[1:, 5]
        Y = np.stack([steering_real_vec, xdot2, ydot2, thetadot2], axis=1)
        X_chunks.append(X)
        Y_chunks.append(Y)
        used_ids.append(id_)

    if verbose:
        print()
        print(f"Successfully loaded: {len(used_ids)}/{total}")
        print(f"Skipped (error or short traj): {skipped}/{total}")

    if not X_chunks:
        return np.empty((0, 5), dtype=float), np.empty((0, 4), dtype=float), []
    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    return X_all, Y_all, used_ids


def build_dataset_filtered(
    data_dir: str = DATA_DIR, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build dataset filtering out trajectories with physically impossible radii.
    Only includes trajectories where radius >= L/2.
    """
    if verbose:
        print("Scanning directory for matching file sets...")
    ids = find_matching_ids(data_dir, verbose=verbose)
    total = len(ids)
    if verbose:
        print(f"Found {total} matching file sets (traj_vels + plan + circle)")

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    used_ids: List[int] = []
    skipped = 0
    skipped_invalid_radius = 0

    for idx, id_ in enumerate(ids, 1):
        if verbose and (idx % 10 == 0 or idx == total):
            print(f"Processing {idx}/{total} (ID: {id_})...", end="\r")

        traj_vels, plan, steering_real, _ = load_traj_data(id_, data_dir=data_dir)
        if traj_vels is None or plan is None or steering_real is None:
            skipped += 1
            continue
        if traj_vels.shape[0] < 2:
            skipped += 1
            continue

        # Skip if steering_real is NaN (invalid radius)
        if np.isnan(steering_real):
            skipped_invalid_radius += 1
            continue

        xdot1 = traj_vels[:-1, 3]
        ydot1 = traj_vels[:-1, 4]
        thetadot1 = traj_vels[:-1, 5]
        steering_u = np.full_like(xdot1, plan[0], dtype=float)
        velocity_u = np.full_like(xdot1, plan[1], dtype=float)
        X = np.stack([xdot1, ydot1, thetadot1, steering_u, velocity_u], axis=1)
        steering_real_vec = np.full_like(xdot1, steering_real, dtype=float)
        xdot2 = traj_vels[1:, 3]
        ydot2 = traj_vels[1:, 4]
        thetadot2 = traj_vels[1:, 5]
        Y = np.stack([steering_real_vec, xdot2, ydot2, thetadot2], axis=1)
        X_chunks.append(X)
        Y_chunks.append(Y)
        used_ids.append(id_)

    if verbose:
        print()
        print(f"Successfully loaded: {len(used_ids)}/{total}")
        print(f"Skipped (error or short traj): {skipped}/{total}")
        print(
            f"Skipped (invalid radius < {MIN_PHYSICAL_RADIUS:.3f}m): {skipped_invalid_radius}/{total}"
        )

    if not X_chunks:
        return np.empty((0, 5), dtype=float), np.empty((0, 4), dtype=float), []
    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    return X_all, Y_all, used_ids


def build_dataset_velocity_only(
    data_dir: str = DATA_DIR, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build dataset that only predicts velocity states (no steering angle).
    Includes all trajectories regardless of radius validity.

    Input: [xdot_t, ydot_t, thetadot_t, steering_cmd, velocity_cmd]
    Output: [xdot_t+1, ydot_t+1, thetadot_t+1]
    """
    if verbose:
        print("Scanning directory for matching file sets...")
    ids = find_matching_ids(data_dir, verbose=verbose)
    total = len(ids)
    if verbose:
        print(f"Found {total} matching file sets (traj_vels + plan + circle)")

    X_chunks: List[np.ndarray] = []
    Y_chunks: List[np.ndarray] = []
    used_ids: List[int] = []
    skipped = 0

    for idx, id_ in enumerate(ids, 1):
        if verbose and (idx % 10 == 0 or idx == total):
            print(f"Processing {idx}/{total} (ID: {id_})...", end="\r")

        traj_vels, plan, steering_real, _ = load_traj_data(id_, data_dir=data_dir)
        if traj_vels is None or plan is None:
            skipped += 1
            continue
        if traj_vels.shape[0] < 2:
            skipped += 1
            continue

        xdot1 = traj_vels[:-1, 3]
        ydot1 = traj_vels[:-1, 4]
        thetadot1 = traj_vels[:-1, 5]
        steering_u = np.full_like(xdot1, plan[0], dtype=float)
        velocity_u = np.full_like(xdot1, plan[1], dtype=float)
        X = np.stack([xdot1, ydot1, thetadot1, steering_u, velocity_u], axis=1)

        # Only predict next velocity states (no steering)
        xdot2 = traj_vels[1:, 3]
        ydot2 = traj_vels[1:, 4]
        thetadot2 = traj_vels[1:, 5]
        Y = np.stack([xdot2, ydot2, thetadot2], axis=1)

        X_chunks.append(X)
        Y_chunks.append(Y)
        used_ids.append(id_)

    if verbose:
        print()
        print(f"Successfully loaded: {len(used_ids)}/{total}")
        print(f"Skipped (error or short traj): {skipped}/{total}")

    if not X_chunks:
        return np.empty((0, 5), dtype=float), np.empty((0, 3), dtype=float), []
    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    return X_all, Y_all, used_ids


def _save_txt(
    output_path: str,
    X: np.ndarray,
    Y: np.ndarray,
    ids: List[int],
    velocity_only: bool = False,
) -> None:
    """
    Save dataset to text file.

    Args:
        output_path: Path to save the dataset
        X: Input features array
        Y: Output features array
        ids: List of trajectory IDs
        velocity_only: If True, Y only contains velocity states (no steering)
    """
    # Concatenate X and Y side by side: [X columns | Y columns]
    data = np.concatenate([X, Y], axis=1)

    # Create header describing each column
    header = "Column indices:\n"
    header += "Inputs (X):\n"
    header += "  [0] xdot_t: x velocity at time t (body frame)\n"
    header += "  [1] ydot_t: y velocity at time t (body frame)\n"
    header += "  [2] thetadot_t: angular velocity at time t\n"
    header += "  [3] steering_cmd: commanded steering angle\n"
    header += "  [4] velocity_cmd: commanded velocity\n"
    header += "Outputs (Y):\n"

    if velocity_only:
        header += "  [5] xdot_t+1: x velocity at time t+1 (body frame)\n"
        header += "  [6] ydot_t+1: y velocity at time t+1 (body frame)\n"
        header += "  [7] thetadot_t+1: angular velocity at time t+1\n"
    else:
        header += "  [5] steering_actual: actual steering angle (from circle fit)\n"
        header += "  [6] xdot_t+1: x velocity at time t+1 (body frame)\n"
        header += "  [7] ydot_t+1: y velocity at time t+1 (body frame)\n"
        header += "  [8] thetadot_t+1: angular velocity at time t+1\n"

    header += f"Total samples: {data.shape[0]}"

    # Save the data with space delimiters
    np.savetxt(
        output_path, data, fmt="%.18e", delimiter=" ", header=header, comments="# "
    )

    # Save trajectory IDs to a separate file
    ids_path = output_path.replace(".txt", "_ids.txt")
    with open(ids_path, "w") as f:
        f.write("# Trajectory IDs used in dataset (one per line)\n")
        for id_ in ids:
            f.write(f"{id_}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["all", "original", "filtered", "velocity_only"],
        default="all",
        help="Which dataset(s) to generate",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if args.dataset_type in ["all", "original"]:
        print("\n" + "=" * 80)
        print(
            "Building ORIGINAL dataset (includes all trajectories, may contain NaN)..."
        )
        print("=" * 80)
        X, Y, ids = build_dataset(args.data_dir, verbose=verbose)

        print("\nDataset shape:")
        print(f"  X (inputs): {X.shape}")
        print(f"  Y (outputs): {Y.shape}")
        print(f"  Total samples: {X.shape[0]}")

        _save_txt(args.output, X, Y, ids)
        print(f"\nSaved dataset to: {args.output}")
        print(f"Saved trajectory IDs to: {args.output.replace('.txt', '_ids.txt')}")

    if args.dataset_type in ["all", "filtered"]:
        print("\n" + "=" * 80)
        print("Building FILTERED dataset (excludes invalid radii)...")
        print("=" * 80)
        X, Y, ids = build_dataset_filtered(args.data_dir, verbose=verbose)

        print("\nDataset shape:")
        print(f"  X (inputs): {X.shape}")
        print(f"  Y (outputs): {Y.shape}")
        print(f"  Total samples: {X.shape[0]}")

        _save_txt(FILTERED_OUTPUT, X, Y, ids)
        print(f"\nSaved filtered dataset to: {FILTERED_OUTPUT}")
        print(f"Saved trajectory IDs to: {FILTERED_OUTPUT.replace('.txt', '_ids.txt')}")

    if args.dataset_type in ["all", "velocity_only"]:
        print("\n" + "=" * 80)
        print("Building VELOCITY-ONLY dataset (no steering prediction)...")
        print("=" * 80)
        X, Y, ids = build_dataset_velocity_only(args.data_dir, verbose=verbose)

        print("\nDataset shape:")
        print(f"  X (inputs): {X.shape}")
        print(f"  Y (outputs): {Y.shape}")
        print(f"  Total samples: {X.shape[0]}")

        _save_txt(VELOCITY_ONLY_OUTPUT, X, Y, ids, velocity_only=True)
        print(f"\nSaved velocity-only dataset to: {VELOCITY_ONLY_OUTPUT}")
        print(
            f"Saved trajectory IDs to: {VELOCITY_ONLY_OUTPUT.replace('.txt', '_ids.txt')}"
        )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
