#!/bin/bash
#
# Run all system identification experiments in disciplined sequence
#
# This script runs experiments 0-5 in order, with proper error handling
# and logging. If any experiment fails, the script stops.
#
# Usage:
#   ./run_all_experiments.sh              # Run all experiments from phase 0
#   ./run_all_experiments.sh --start-from 2   # Start from phase 2
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Parse command-line arguments
START_PHASE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            START_PHASE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--start-from PHASE]"
            echo ""
            echo "Options:"
            echo "  --start-from PHASE    Start from the specified phase (0-5)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Available phases:"
            echo "  0: Baseline Sanity Check"
            echo "  1: Trajectory-Based Validation (1a and 1b)"
            echo "  2: Rollout + Pose Loss"
            echo "  3: Friction Rescue"
            echo "  4: Adapter-Only Ablation"
            echo "  5: Adapter + Tiny Residual"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate START_PHASE
if ! [[ "$START_PHASE" =~ ^[0-5]$ ]]; then
    echo "Error: START_PHASE must be between 0 and 5 (got: $START_PHASE)"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_DIR="${PROJECT_ROOT}/experiment_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/all_experiments_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}  System Identification Training - Experiment Sequence${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Master log: $MASTER_LOG"
echo "Starting from: Phase $START_PHASE"
echo ""
if [ "$START_PHASE" -gt 0 ]; then
    echo -e "${YELLOW}Note: Skipping phases 0-$((START_PHASE-1))${NC}"
    echo ""
fi

# Function to run a single experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local config_path=$3

    echo -e "${YELLOW}------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}Experiment $exp_num: $exp_name${NC}"
    echo -e "${YELLOW}------------------------------------------------------------------${NC}"
    echo "Config: $config_path"
    echo "Started: $(date)"
    echo ""

    # Create experiment-specific log
    local exp_log="${LOG_DIR}/exp${exp_num}_$(date +%Y%m%d_%H%M%S).log"

    # Run experiment
    if python "${SCRIPT_DIR}/train.py" --config "$config_path" 2>&1 | tee -a "$exp_log" | tee -a "$MASTER_LOG"; then
        echo "" | tee -a "$MASTER_LOG"
        echo -e "${GREEN}✓ Experiment $exp_num completed successfully${NC}" | tee -a "$MASTER_LOG"
        echo "Finished: $(date)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
        return 0
    else
        echo "" | tee -a "$MASTER_LOG"
        echo -e "${RED}✗ Experiment $exp_num FAILED${NC}" | tee -a "$MASTER_LOG"
        echo "Failed at: $(date)" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
        return 1
    fi
}

# Function to print summary
print_summary() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  Experiment Summary${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""

    for i in "${!COMPLETED_EXPERIMENTS[@]}"; do
        if [ "${COMPLETED_EXPERIMENTS[$i]}" = "1" ]; then
            echo -e "${GREEN}✓ Experiment $i: ${EXPERIMENT_NAMES[$i]}${NC}"
        else
            echo -e "${RED}✗ Experiment $i: ${EXPERIMENT_NAMES[$i]}${NC}"
        fi
    done

    echo ""
    echo "Master log: $MASTER_LOG"
    echo ""
}

# Experiment tracking
declare -a COMPLETED_EXPERIMENTS=(0 0 0 0 0 0 0 0)  # 8 experiments total (0, 1a, 1b, 2-5)
declare -a EXPERIMENT_NAMES=(
    "Baseline Sanity Check"
    "Trajectory Val (Residual)"
    "Trajectory Val (Direct)"
    "Rollout + Pose"
    "Friction Rescue"
    "Adapter Strict"
    "Adapter + Tiny Residual"
)

# Trap to print summary on exit
trap print_summary EXIT

# ============================================================================
# EXPERIMENT 0: Baseline Sanity Check
# ============================================================================

if [ "$START_PHASE" -le 0 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 0: Baseline Validation${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Prove backward compatibility"
    echo "Success: Match previous best numbers within ±5%"
    echo ""
    echo -e "${YELLOW}⚠  CRITICAL: If this fails, STOP and fix before proceeding${NC}"
    echo ""

    if run_experiment 0 "Baseline Sanity Check" \
        "${PROJECT_ROOT}/configs/experiments/exp0_baseline_sanity_check.json"; then
        COMPLETED_EXPERIMENTS[0]=1
    else
        echo -e "${RED}==================================================================${NC}"
        echo -e "${RED}  BASELINE SANITY CHECK FAILED${NC}"
        echo -e "${RED}==================================================================${NC}"
        echo ""
        echo "The baseline experiment failed to reproduce previous results."
        echo "This indicates a potential implementation bug or environment issue."
        echo ""
        echo "DO NOT proceed with other experiments until this is fixed!"
        echo ""
        exit 1
    fi

    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}  Baseline validated! Proceeding with improvements...${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 0: Baseline Validation${NC}"
    echo ""
fi

# ============================================================================
# EXPERIMENT 1: Trajectory Validation
# ============================================================================

if [ "$START_PHASE" -le 1 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 1: Trajectory-Based Validation${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Fix validation selection (no train/val leakage)"
    echo "Configs: Residual-only and Direct models"
    echo ""

    if run_experiment 1a "Trajectory Validation (Residual)" \
        "${PROJECT_ROOT}/configs/experiments/exp1_traj_val_residual.json"; then
        COMPLETED_EXPERIMENTS[1]=1
    fi

    if run_experiment 1b "Trajectory Validation (Direct)" \
        "${PROJECT_ROOT}/configs/experiments/exp1_traj_val_direct.json"; then
        COMPLETED_EXPERIMENTS[2]=1
    fi

    echo -e "${GREEN}Phase 1 complete! Trajectory validation tested.${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 1: Trajectory-Based Validation${NC}"
    echo ""
fi

# ============================================================================
# EXPERIMENT 2: Rollout + Pose Loss
# ============================================================================

if [ "$START_PHASE" -le 2 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 2: Rollout + Pose Loss${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Validate rollout pipeline, improve position accuracy"
    echo "Settings: Conservative (rollout 0.5, teacher forcing 0.2, pose 0.05)"
    echo ""

    if run_experiment 2 "Rollout + Pose (Conservative)" \
        "${PROJECT_ROOT}/configs/experiments/exp2_rollout_pose_residual.json"; then
        COMPLETED_EXPERIMENTS[3]=1
    fi

    echo -e "${GREEN}Phase 2 complete! Rollout pipeline tested.${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 2: Rollout + Pose Loss${NC}"
    echo ""
fi

# ============================================================================
# EXPERIMENT 3: Friction Rescue
# ============================================================================

if [ "$START_PHASE" -le 3 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 3: Friction-Only Rescue${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Test if friction-only can be rescued with better parameterization"
    echo "Changes: sigmoid_range friction [0.25, 3.0], prior 0.01, rollout+pose"
    echo ""
    echo -e "${YELLOW}⚠  IMPORTANT: After this run, inspect friction_k values!${NC}"
    echo ""

    if run_experiment 3 "Friction Rescue (Sigmoid Range)" \
        "${PROJECT_ROOT}/configs/experiments/exp3_friction_rescue.json"; then
        COMPLETED_EXPERIMENTS[4]=1

        echo ""
        echo -e "${YELLOW}------------------------------------------------------------------${NC}"
        echo -e "${YELLOW}  POST-EXPERIMENT DIAGNOSTIC CHECKLIST${NC}"
        echo -e "${YELLOW}------------------------------------------------------------------${NC}"
        echo ""
        echo "Check friction_k values in logs/tensorboard:"
        echo ""
        echo "  - Saturates at k_min=0.25  → Need lower k_min (try 0.15)"
        echo "  - Saturates at k_max=3.0   → Need higher k_max (try 4-5)"
        echo "  - Stays near 1.0, static   → Prior too strong (try 0.001)"
        echo "  - Varies widely, unstable  → Prior too weak (try 0.05)"
        echo "  - In [0.5, 1.5], improves  → SUCCESS!"
        echo ""
    fi

    echo -e "${GREEN}Phase 3 complete! Friction rescue tested.${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 3: Friction-Only Rescue${NC}"
    echo ""
fi

# ============================================================================
# EXPERIMENT 4: Adapter-Only Strict Ablation
# ============================================================================

if [ "$START_PHASE" -le 4 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 4: Adapter-Only Ablation${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Evidence for 'adapter alone is insufficient'"
    echo "Expectation: May plateau - this is valuable ablation evidence"
    echo ""

    if run_experiment 4 "Adapter-Only Strict Ablation" \
        "${PROJECT_ROOT}/configs/experiments/exp4_adapter_strict_ablation.json"; then
        COMPLETED_EXPERIMENTS[5]=1
    fi

    echo -e "${GREEN}Phase 4 complete! Adapter-only ablation tested.${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 4: Adapter-Only Ablation${NC}"
    echo ""
fi

# ============================================================================
# EXPERIMENT 5: Adapter + Tiny Residual
# ============================================================================

if [ "$START_PHASE" -le 5 ]; then
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}  PHASE 5: Adapter + Tiny Residual${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Purpose: Test if plant mismatch is representable by control remapping"
    echo "Key: Strong L2 penalty (0.05) keeps residual small"
    echo ""
    echo "Interpretation:"
    echo "  - If Exp 4 fails but this succeeds → adapter alone insufficient"
    echo "  - If residual grows large → this is just 'residual model'"
    echo "  - If residual stays tiny & good → ideal compromise"
    echo ""

    if run_experiment 5 "Adapter + Tiny Residual" \
        "${PROJECT_ROOT}/configs/experiments/exp5_adapter_tiny_residual.json"; then
        COMPLETED_EXPERIMENTS[6]=1

        echo ""
        echo -e "${YELLOW}------------------------------------------------------------------${NC}"
        echo -e "${YELLOW}  POST-EXPERIMENT DIAGNOSTIC CHECKLIST${NC}"
        echo -e "${YELLOW}------------------------------------------------------------------${NC}"
        echo ""
        echo "Check residual magnitudes in logs:"
        echo ""
        echo "  - Residual ~0.01, good perf   → Ideal (tiny but effective)"
        echo "  - Residual >0.1, good perf    → Just 'residual model'"
        echo "  - Residual ~0.01, bad perf    → Neither adapter nor tiny residual helps"
        echo ""
        echo "Compare Exp 4 vs Exp 5 eval_pos_mse:"
        echo ""
        echo "  - Large improvement → Plant mismatch NOT representable by adapter"
        echo "  - Small improvement → Adapter is nearly sufficient"
        echo ""
    fi

    echo -e "${GREEN}Phase 5 complete! All experiments finished!${NC}"
    echo ""
    sleep 2
else
    echo -e "${YELLOW}Skipping Phase 5: Adapter + Tiny Residual${NC}"
    echo ""
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}  ALL EXPERIMENTS COMPLETE${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""
echo "Finished: $(date)"
echo ""
echo "Results are in:"
for i in {0..5}; do
    if [ $i -eq 1 ]; then
        echo "  experiments/exp1_traj_val/residual_only/"
        echo "  experiments/exp1_traj_val/direct/"
    else
        exp_folder=$(ls -d experiments/exp${i}_* 2>/dev/null | head -1)
        if [ -n "$exp_folder" ]; then
            echo "  $exp_folder"
        fi
    fi
done
echo ""
echo "Logs are in: $LOG_DIR"
echo "Master log: $MASTER_LOG"
echo ""
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  NEXT STEPS${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo ""
echo "1. Review eval_pos_mse and eval_vel_mse across experiments"
echo "2. Check friction_k values in Exp 3 (friction rescue)"
echo "3. Check residual magnitudes in Exp 5 (adapter + tiny residual)"
echo "4. Compare Exp 4 vs Exp 5 to assess adapter sufficiency"
echo "5. If rollout+pose helped (Exp 2), try more aggressive settings"
echo "6. Document findings and successful patterns"
echo ""
echo "See EXPERIMENT_GUIDE.md for detailed analysis guidance."
echo ""
