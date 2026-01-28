#!/usr/bin/env python
"""
Export trained sysid models to TorchScript for C++ inference.

Usage:
    python export_torchscript.py --exp_dir /path/to/experiment --output_dir /path/to/output --dt 0.05
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mushr_mujoco_sysid.config_utils import populate_config_defaults
from mushr_mujoco_sysid.model_factory import build_model
from mushr_mujoco_sysid.models.system_models import ControlAdapter
from mushr_mujoco_sysid.utils import load_standardizers_json
from mushr_mujoco_sysid.deploy.mlp_manual_jac import MLPManualJac
from mushr_mujoco_sysid.deploy.deploy_modules import (
    DirectDeployModule,
    StructuredAuxDeployModule,
)


def load_experiment(exp_dir: str):
    config_path = os.path.join(exp_dir, "config.json")
    ckpt_path = os.path.join(exp_dir, "best.pt")
    std_path = os.path.join(exp_dir, "standardizers.json")

    for p, name in [(config_path, "config.json"), (ckpt_path, "best.pt"), (std_path, "standardizers.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name} in {exp_dir}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg = populate_config_defaults(cfg)

    input_std, target_std = load_standardizers_json(std_path)

    model = build_model(cfg, device="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return cfg, model, input_std, target_std


def convert_adapter_nets(adapter: ControlAdapter):
    steer_net = MLPManualJac.from_mlp(adapter.steer_net)
    acc_net = MLPManualJac.from_mlp(adapter.acc_net)
    
    adapter_config = {
        "include_speed_mag_steer": adapter.include_speed_mag_steer,
        "include_speed_sign_steer": adapter.include_speed_sign_steer,
        "include_speed_mag_acc": adapter.include_speed_mag_acc,
        "include_speed_sign_acc": adapter.include_speed_sign_acc,
        "include_vy_acc": adapter.include_vy_acc,
        "include_w_acc": adapter.include_w_acc,
        "steer_output_scale": adapter.steer_output_scale,
        "acc_output_scale": adapter.acc_output_scale,
    }
    
    return steer_net, acc_net, adapter_config


def export_direct_model(model, input_std, target_std, dt, output_path, dtype=torch.float64):
    input_mean = torch.tensor(input_std.mean, dtype=dtype)
    input_std_t = torch.tensor(input_std.std, dtype=dtype)
    target_mean = torch.tensor(target_std.mean, dtype=dtype)
    target_std_t = torch.tensor(target_std.std, dtype=dtype)

    net = MLPManualJac.from_mlp(model.net)

    has_adapter = model.control_adapter is not None
    adapter_steer = None
    adapter_acc = None
    adapter_config = None

    if has_adapter:
        adapter_steer, adapter_acc, adapter_config = convert_adapter_nets(model.control_adapter)

    deploy = DirectDeployModule(
        net=net,
        input_mean=input_mean,
        input_std=input_std_t,
        target_mean=target_mean,
        target_std=target_std_t,
        dt=dt,
        has_adapter=has_adapter,
        adapter_steer_net=adapter_steer,
        adapter_acc_net=adapter_acc,
        adapter_config=adapter_config,
    )
    deploy.eval()

    scripted = torch.jit.script(deploy)
    # Don't freeze - it strips the forward_with_jacobian exported method
    scripted.save(output_path)

    print(f"Exported DirectDeployModule to {output_path}")
    return deploy


def export_structured_aux(model, input_std, target_std, dt, output_path, dtype=torch.float64):
    input_mean = torch.tensor(input_std.mean, dtype=dtype)
    input_std_t = torch.tensor(input_std.std, dtype=dtype)
    target_mean = torch.tensor(target_std.mean, dtype=dtype)
    target_std_t = torch.tensor(target_std.std, dtype=dtype)

    has_adapter = model.control_adapter is not None
    adapter_steer = None
    adapter_acc = None
    adapter_config = None

    if has_adapter:
        adapter_steer, adapter_acc, adapter_config = convert_adapter_nets(model.control_adapter)

    has_friction = model.friction_net is not None
    friction_net = None
    friction_config = None

    if has_friction:
        friction_net = MLPManualJac.from_mlp(model.friction_net)
        friction_config = {
            "friction_use_dt": model.friction_use_dt,
            "friction_use_delta": model.friction_use_delta,
            "friction_use_vy": model.friction_use_vy,
            "friction_param_mode": model.friction_param_mode,
            "friction_k_min": model.friction_k_min,
            "friction_k_max": model.friction_k_max,
        }

    has_residual = model.residual_net is not None
    residual_net = None

    if has_residual:
        residual_net = MLPManualJac.from_mlp(model.residual_net)

    deploy = StructuredAuxDeployModule(
        input_mean=input_mean,
        input_std=input_std_t,
        target_mean=target_mean,
        target_std=target_std_t,
        dt=dt,
        L=model.plant.L,
        has_adapter=has_adapter,
        adapter_steer_net=adapter_steer,
        adapter_acc_net=adapter_acc,
        adapter_config=adapter_config,
        has_friction=has_friction,
        friction_net=friction_net,
        friction_config=friction_config,
        has_residual=has_residual,
        residual_net=residual_net,
    )
    deploy.eval()

    scripted = torch.jit.script(deploy)
    # Don't freeze - it strips the forward_with_jacobian exported method
    scripted.save(output_path)

    print(f"Exported StructuredAuxDeployModule to {output_path}")
    return deploy


def validate_export(deploy_module, is_structured, n_tests=10, dtype=torch.float64):
    print("\nValidating export...")

    rng = np.random.default_rng(42)

    for i in range(n_tests):
        xd0_raw = torch.tensor(rng.normal(0, 1, 3), dtype=dtype)
        ut_raw = torch.tensor(rng.normal(0, 0.5, 2), dtype=dtype)
        
        with torch.no_grad():
            if is_structured:
                ut_eff, k, residual = deploy_module(xd0_raw, ut_raw)
                result = deploy_module.forward_with_jacobian(xd0_raw, ut_raw)
                assert torch.allclose(ut_eff, result[0], atol=1e-6)
            else:
                y_deploy = deploy_module(xd0_raw, ut_raw)
                y_jac, Jx, Ju = deploy_module.forward_with_jacobian(xd0_raw, ut_raw)
                assert torch.allclose(y_deploy, y_jac, atol=1e-6)
    
    print(f"  Forward/Jacobian consistency: PASSED ({n_tests} tests)")


def validate_jacobians_numerical(deploy_module, is_structured, n_tests=5, eps=1e-5, dtype=torch.float64):
    print("Validating Jacobians against numerical derivatives...")

    rng = np.random.default_rng(123)
    max_err_x = 0.0
    max_err_u = 0.0

    for i in range(n_tests):
        xd0_raw = torch.tensor(rng.normal(0, 1, 3), dtype=dtype)
        ut_raw = torch.tensor(rng.normal(0, 0.5, 2), dtype=dtype)
        
        with torch.no_grad():
            if is_structured:
                result = deploy_module.forward_with_jacobian(xd0_raw, ut_raw)
                Jx_analytic = result[3]
                Ju_analytic = result[4]
            else:
                _, Jx_analytic, Ju_analytic = deploy_module.forward_with_jacobian(xd0_raw, ut_raw)
        
        Jx_numerical = torch.zeros_like(Jx_analytic)
        for j in range(3):
            xd0_plus = xd0_raw.clone()
            xd0_plus[j] += eps
            xd0_minus = xd0_raw.clone()
            xd0_minus[j] -= eps
            
            with torch.no_grad():
                if is_structured:
                    y_plus = deploy_module(xd0_plus, ut_raw)[0]
                    y_minus = deploy_module(xd0_minus, ut_raw)[0]
                else:
                    y_plus = deploy_module(xd0_plus, ut_raw)
                    y_minus = deploy_module(xd0_minus, ut_raw)
            
            Jx_numerical[:, j] = (y_plus - y_minus) / (2 * eps)
        
        Ju_numerical = torch.zeros_like(Ju_analytic)
        for j in range(2):
            ut_plus = ut_raw.clone()
            ut_plus[j] += eps
            ut_minus = ut_raw.clone()
            ut_minus[j] -= eps
            
            with torch.no_grad():
                if is_structured:
                    y_plus = deploy_module(xd0_raw, ut_plus)[0]
                    y_minus = deploy_module(xd0_raw, ut_minus)[0]
                else:
                    y_plus = deploy_module(xd0_raw, ut_plus)
                    y_minus = deploy_module(xd0_raw, ut_minus)
            
            Ju_numerical[:, j] = (y_plus - y_minus) / (2 * eps)
        
        err_x = (Jx_analytic - Jx_numerical).abs().max().item()
        err_u = (Ju_analytic - Ju_numerical).abs().max().item()
        max_err_x = max(max_err_x, err_x)
        max_err_u = max(max_err_u, err_u)
    
    # Use dtype-appropriate tolerance (float32 has ~1e-6 precision)
    tol = 1e-4 if dtype == torch.float64 else 5e-2
    print(f"  Jx max error: {max_err_x:.2e} [{'PASSED' if max_err_x < tol else 'FAILED'}]")
    print(f"  Ju max error: {max_err_u:.2e} [{'PASSED' if max_err_u < tol else 'FAILED'}]")


def main():
    parser = argparse.ArgumentParser(description="Export trained sysid models to TorchScript")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.exp_dir
    os.makedirs(output_dir, exist_ok=True)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print("=" * 70)
    print("TorchScript Export")
    print("=" * 70)
    print(f"Experiment: {args.exp_dir}")
    print(f"Output: {output_dir}")
    print(f"dt: {args.dt}")
    print(f"dtype: {args.dtype}")

    cfg, model, input_std, target_std = load_experiment(args.exp_dir)
    model_type = cfg["model"].get("type", "structured").lower()

    print(f"Model type: {model_type}")

    if model_type == "direct":
        output_path = os.path.join(output_dir, "direct_model.ts.pt")
        deploy = export_direct_model(model, input_std, target_std, args.dt, output_path, dtype)
        is_structured = False
    else:
        output_path = os.path.join(output_dir, "structured_aux.ts.pt")
        deploy = export_structured_aux(model, input_std, target_std, args.dt, output_path, dtype)
        is_structured = True

    if not args.skip_validation:
        validate_export(deploy, is_structured, dtype=dtype)
        validate_jacobians_numerical(deploy, is_structured, dtype=dtype)

    # Save export metadata for C++ inference
    metadata = {
        "dtype": args.dtype,
        "dt": args.dt,
        "model_type": model_type,
        "model_file": os.path.basename(output_path),
    }
    metadata_path = os.path.join(output_dir, "export_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved export metadata to {metadata_path}")

    print("=" * 70)
    print(f"Export complete! Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
