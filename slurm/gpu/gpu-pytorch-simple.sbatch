#!/bin/bash
#SBATCH --job-name=gpu-pytorch-simple
#SBATCH --output=gpu-pytorch-simple-%j.out
#SBATCH --error=gpu-pytorch-simple-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# Workload: GPU PyTorch Simple Training
# Description: Simple neural network classification with synthetic data using PyTorch (GPU)
# Type: GPU

echo "Starting GPU PyTorch workload on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

WORKSPACE_ROOT="$SLURM_SUBMIT_DIR"
while [[ "$WORKSPACE_ROOT" != "/" ]]; do
    if [[ -d "$WORKSPACE_ROOT/gpu_smr" ]]; then
        break
    fi
    WORKSPACE_ROOT="$(dirname "$WORKSPACE_ROOT")"
done

if [[ ! -d "$WORKSPACE_ROOT/gpu_smr" ]]; then
    echo "ERROR: Could not find workspace root"
    exit 1
fi

cd "$WORKSPACE_ROOT" || exit 1

if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Installing with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118 --quiet
    echo "PyTorch installed successfully"
fi

echo "GPU Info:"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU Count: {torch.cuda.device_count()}'); print(f'  GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

python3 gpu_smr/pytorch/training/test.py
