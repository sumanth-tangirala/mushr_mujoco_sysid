#!/bin/bash
# Run all 8 v3 allstars experiments on 8 separate GPUs using tmux
# Usage: bash scripts/run_v3_allstars_tmux.sh

SESSION_NAME="v3_allstars"

# Array of config files and their assigned GPUs
declare -a CONFIGS=(
    "configs/v3_allstars_controls_vary/v3A_struct_exp2_replay_seed4.json"
    "configs/v3_allstars_controls_vary/v3B_struct_h10_tf0_seed4.json"
    "configs/v3_allstars_controls_vary/v3C_struct_h20_tf0_seed4.json"
    "configs/v3_allstars_controls_vary/v3D_struct_h20_tf0_seed2.json"
    "configs/v3_allstars_controls_vary/v3E_struct_h20_tf0_resl2_0p01_seed4.json"
    "configs/v3_allstars_controls_vary/v3F_struct_h20_tf0_pose_xytheta_seed4.json"
    "configs/v3_allstars_controls_vary/v3G_direct_no_adapter_h20_tf0_seed4.json"
    "configs/v3_allstars_controls_vary/v3H_direct_with_adapter_h20_tf0_seed4.json"
)

declare -a EXPERIMENT_NAMES=(
    "v3A_exp2_replay"
    "v3B_h10_tf0"
    "v3C_h20_tf0"
    "v3D_h20_tf0_s2"
    "v3E_h20_resl2"
    "v3F_h20_theta"
    "v3G_direct_no_adp"
    "v3H_direct_adp"
)

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

echo "Creating tmux session: $SESSION_NAME"
echo "Running 8 experiments on GPUs 0-7"
echo ""

# Create a new detached tmux session with the first window
tmux new-session -d -s $SESSION_NAME -n "${EXPERIMENT_NAMES[0]}"

# Run first experiment in the first window
GPU=0
CONFIG="${CONFIGS[0]}"
echo "Window 0 (GPU $GPU): ${EXPERIMENT_NAMES[0]}"
tmux send-keys -t $SESSION_NAME:0 "export CUDA_VISIBLE_DEVICES=$GPU" C-m
tmux send-keys -t $SESSION_NAME:0 "python scripts/train.py --config $CONFIG" C-m

# Create additional windows for the remaining experiments
for i in {1..7}; do
    GPU=$i
    CONFIG="${CONFIGS[$i]}"
    WINDOW_NAME="${EXPERIMENT_NAMES[$i]}"

    echo "Window $i (GPU $GPU): $WINDOW_NAME"

    # Create new window
    tmux new-window -t $SESSION_NAME:$i -n "$WINDOW_NAME"

    # Set GPU and run training
    tmux send-keys -t $SESSION_NAME:$i "export CUDA_VISIBLE_DEVICES=$GPU" C-m
    tmux send-keys -t $SESSION_NAME:$i "python scripts/train.py --config $CONFIG" C-m
done

echo ""
echo "All 8 experiments started in tmux session '$SESSION_NAME'"
echo ""
echo "Commands:"
echo "  Attach to session:    tmux attach-session -t $SESSION_NAME"
echo "  List windows:         tmux list-windows -t $SESSION_NAME"
echo "  Switch windows:       Ctrl+b then 0-7 (or Ctrl+b n/p for next/previous)"
echo "  Detach from session:  Ctrl+b then d"
echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
echo ""
echo "Window layout:"
for i in {0..7}; do
    echo "  Window $i: GPU $i - ${EXPERIMENT_NAMES[$i]}"
done
