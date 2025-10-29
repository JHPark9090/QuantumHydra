#!/bin/bash
#SBATCH -A m4138_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=compare_all_models
#SBATCH --output=logs/compare_all_models_%j.out
#SBATCH --error=logs/compare_all_models_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@example.com

################################################################################
# COMPREHENSIVE MODEL COMPARISON
################################################################################
#
# Compares four state-space models:
#   1. Quantum Hydra (Superposition) - Option A
#   2. Quantum Hydra (Hybrid) - Option B
#   3. True Classical Hydra - Corrected baseline
#   4. True Classical Mamba - Additional baseline
#
# On PhysioNet EEG Motor Imagery dataset
#
################################################################################

module load python
module load cuda
mkdir -p logs
source activate ./conda-envs/qml_env

echo "========================================================================"
echo "COMPREHENSIVE MODEL COMPARISON"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "========================================================================"

# Configuration
SEED=2024
N_QUBITS=4
QLCU_LAYERS=2
HIDDEN_DIM=64
D_MODEL=128
D_STATE=16
N_EPOCHS=50
BATCH_SIZE=16
LR=1e-3
SAMPLE_SIZE=10
SAMPLING_FREQ=100

echo ""
echo "Configuration:"
echo "  Quantum Models:"
echo "    - Qubits: $N_QUBITS"
echo "    - QLCU Layers: $QLCU_LAYERS"
echo "  Classical Models:"
echo "    - Hydra Hidden Dim: $HIDDEN_DIM"
echo "    - Mamba d_model: $D_MODEL"
echo "    - Mamba d_state: $D_STATE"
echo "  Training:"
echo "    - Epochs: $N_EPOCHS"
echo "    - Batch Size: $BATCH_SIZE"
echo "    - Learning Rate: $LR"
echo "  Data:"
echo "    - Sample Size: $SAMPLE_SIZE subjects"
echo "    - Sampling Freq: $SAMPLING_FREQ Hz"
echo "    - Seed: $SEED"
echo "========================================================================"

echo ""
echo "Running comprehensive comparison..."
echo ""

python compare_all_models.py \
    --n-qubits=$N_QUBITS \
    --qlcu-layers=$QLCU_LAYERS \
    --hidden-dim=$HIDDEN_DIM \
    --d-model=$D_MODEL \
    --d-state=$D_STATE \
    --n-epochs=$N_EPOCHS \
    --batch-size=$BATCH_SIZE \
    --lr=$LR \
    --sample-size=$SAMPLE_SIZE \
    --sampling-freq=$SAMPLING_FREQ \
    --seed=$SEED \
    --device=cuda \
    2>&1 | tee logs/comparison_${SLURM_JOB_ID}.log

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Comparison Complete"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Finished: $(date)"
echo "========================================================================"

# Save job info
echo "Job ID: $SLURM_JOB_ID" > comparison_info_${SLURM_JOB_ID}.txt
echo "Seed: $SEED" >> comparison_info_${SLURM_JOB_ID}.txt
echo "Exit code: $EXIT_CODE" >> comparison_info_${SLURM_JOB_ID}.txt
echo "Finished: $(date)" >> comparison_info_${SLURM_JOB_ID}.txt

exit $EXIT_CODE
