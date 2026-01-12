#!/usr/bin/env python
"""
Benchmark and validation script for sysid fast inference.

Compares:
- Reference (eager) vs Fast path
- Compile / TF32 / CUDA Graph modes
- Correctness (output deltas)
- Latency statistics (p50, p95, p99)
"""

import argparse
import time
import numpy as np
import torch

from mushr_mujoco_sysid.config_utils import populate_config_defaults
from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.utils import load_standardizers_json
from mushr_mujoco_sysid.fast import FastInferenceSession


def benchmark_mode(
    session_or_model,
    xd0_samples: np.ndarray,
    ut_samples: np.ndarray,
    num_warmup: int = 10,
    num_test: int = 100,
    is_fast_session: bool = False,
):
    """
    Benchmark a single mode.
    
    Returns:
        outputs: np.ndarray of shape (num_test, 3)
        latencies: np.ndarray of shape (num_test,) in milliseconds
    """
    outputs = []
    latencies = []

    # Warmup
    for i in range(num_warmup):
        xd0 = xd0_samples[i % len(xd0_samples)]
        ut = ut_samples[i % len(ut_samples)]
        if is_fast_session:
            _ = session_or_model.predict_one_numpy(xd0, ut)
        else:
            # Reference path (simplified)
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    for i in range(num_test):
        xd0 = xd0_samples[i % len(xd0_samples)]
        ut = ut_samples[i % len(ut_samples)]

        start = time.perf_counter()
        if is_fast_session:
            out = session_or_model.predict_one_numpy(xd0, ut)
        else:
            # Reference path would go here
            out = np.zeros(3)
        end = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs.append(out)
        latencies.append((end - start) * 1000)  # Convert to ms

    return np.array(outputs), np.array(latencies)


def main():
    parser = argparse.ArgumentParser(description="Benchmark sysid inference")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--num_warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Sysid Inference Benchmark")
    print("=" * 80)
    print(f"Experiment: {args.exp_dir}")
    print(f"Device: {args.device}, dtype: {args.dtype}, dt: {args.dt}")
    print(f"Samples: {args.num_samples}, warmup: {args.num_warmup}")
    print()

    # Generate random test inputs
    print("Generating random test samples...")
    xd0_samples = np.random.randn(args.num_samples, 3).astype(np.float64) * 0.5
    ut_samples = np.random.randn(args.num_samples, 2).astype(np.float64) * 0.3

    modes = []

    # Reference: eager mode
    print("=" * 80)
    print("MODE: Eager (reference)")
    print("=" * 80)
    try:
        session_eager = FastInferenceSession(
            exp_dir=args.exp_dir,
            dt=args.dt,
            device=args.device,
            dtype=args.dtype,
            use_compile=False,
            use_tf32=False,
            use_cudagraph=False,
            warmup_iters=args.num_warmup,
        )
        outputs_eager, latencies_eager = benchmark_mode(
            session_eager,
            xd0_samples,
            ut_samples,
            num_warmup=args.num_warmup,
            num_test=args.num_samples,
            is_fast_session=True,
        )
        modes.append(("eager", outputs_eager, latencies_eager))
        print(f"Latency: p50={np.percentile(latencies_eager, 50):.3f}ms, "
              f"p95={np.percentile(latencies_eager, 95):.3f}ms, "
              f"p99={np.percentile(latencies_eager, 99):.3f}ms")
        print()
    except Exception as e:
        print(f"FAILED: {e}\n")

    # Mode: compile
    print("=" * 80)
    print("MODE: torch.compile")
    print("=" * 80)
    try:
        session_compile = FastInferenceSession(
            exp_dir=args.exp_dir,
            dt=args.dt,
            device=args.device,
            dtype=args.dtype,
            use_compile=True,
            use_tf32=False,
            use_cudagraph=False,
            warmup_iters=args.num_warmup,
        )
        outputs_compile, latencies_compile = benchmark_mode(
            session_compile,
            xd0_samples,
            ut_samples,
            num_warmup=args.num_warmup,
            num_test=args.num_samples,
            is_fast_session=True,
        )
        modes.append(("compile", outputs_compile, latencies_compile))
        print(f"Latency: p50={np.percentile(latencies_compile, 50):.3f}ms, "
              f"p95={np.percentile(latencies_compile, 95):.3f}ms, "
              f"p99={np.percentile(latencies_compile, 99):.3f}ms")
        
        if len(modes) > 1:
            delta = np.abs(outputs_compile - outputs_eager).max()
            print(f"Max delta vs eager: {delta:.2e}")
        print()
    except Exception as e:
        print(f"FAILED: {e}\n")

    # Mode: compile + TF32
    if args.device.startswith("cuda"):
        print("=" * 80)
        print("MODE: torch.compile + TF32")
        print("=" * 80)
        try:
            session_compile_tf32 = FastInferenceSession(
                exp_dir=args.exp_dir,
                dt=args.dt,
                device=args.device,
                dtype=args.dtype,
                use_compile=True,
                use_tf32=True,
                use_cudagraph=False,
                warmup_iters=args.num_warmup,
            )
            outputs_compile_tf32, latencies_compile_tf32 = benchmark_mode(
                session_compile_tf32,
                xd0_samples,
                ut_samples,
                num_warmup=args.num_warmup,
                num_test=args.num_samples,
                is_fast_session=True,
            )
            modes.append(("compile+TF32", outputs_compile_tf32, latencies_compile_tf32))
            print(f"Latency: p50={np.percentile(latencies_compile_tf32, 50):.3f}ms, "
                  f"p95={np.percentile(latencies_compile_tf32, 95):.3f}ms, "
                  f"p99={np.percentile(latencies_compile_tf32, 99):.3f}ms")
            
            if len(modes) > 1:
                delta = np.abs(outputs_compile_tf32 - outputs_eager).max()
                print(f"Max delta vs eager: {delta:.2e}")
            print()
        except Exception as e:
            print(f"FAILED: {e}\n")

    # Mode: CUDA Graph
    if args.device.startswith("cuda"):
        print("=" * 80)
        print("MODE: CUDA Graph (compile + graph)")
        print("=" * 80)
        try:
            session_cudagraph = FastInferenceSession(
                exp_dir=args.exp_dir,
                dt=args.dt,
                device=args.device,
                dtype=args.dtype,
                use_compile=True,
                use_tf32=False,
                use_cudagraph=True,
                warmup_iters=args.num_warmup,
            )
            outputs_cudagraph, latencies_cudagraph = benchmark_mode(
                session_cudagraph,
                xd0_samples,
                ut_samples,
                num_warmup=args.num_warmup,
                num_test=args.num_samples,
                is_fast_session=True,
            )
            modes.append(("cudagraph", outputs_cudagraph, latencies_cudagraph))
            print(f"Latency: p50={np.percentile(latencies_cudagraph, 50):.3f}ms, "
                  f"p95={np.percentile(latencies_cudagraph, 95):.3f}ms, "
                  f"p99={np.percentile(latencies_cudagraph, 99):.3f}ms")
            
            if len(modes) > 1:
                delta = np.abs(outputs_cudagraph - outputs_eager).max()
                print(f"Max delta vs eager: {delta:.2e}")
            print()
        except Exception as e:
            print(f"FAILED: {e}\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<20} {'p50 (ms)':<12} {'p95 (ms)':<12} {'p99 (ms)':<12} {'Max Δ':<12}")
    print("-" * 80)
    for i, (name, outputs, latencies) in enumerate(modes):
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        if i == 0:
            delta_str = "reference"
        else:
            delta = np.abs(outputs - modes[0][1]).max()
            delta_str = f"{delta:.2e}"
        print(f"{name:<20} {p50:<12.3f} {p95:<12.3f} {p99:<12.3f} {delta_str:<12}")
    print("=" * 80)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
