# Quantum vs Classical State-Space Models - Ablation Study Guide

> Complete guide for conducting ablation studies comparing Quantum Hydra, Quantum Mamba, and their classical counterparts

## 🎯 Overview

You now have **6 state-space models** ready for comprehensive ablation studies:

### Quantum Models (4 total)
1. **Quantum Hydra (Superposition)** - Option A
2. **Quantum Hydra (Hybrid)** - Option B
3. **Quantum Mamba (Superposition)** - Option A [NEW]
4. **Quantum Mamba (Hybrid)** - Option B [NEW]

### Classical Baselines (2 total)
5. **True Classical Hydra** - Corrected baseline
6. **True Classical Mamba** - State-of-the-art SSM

---

## 📊 Model Files Summary

| Model | File | Size | Type | Status |
|-------|------|------|------|--------|
| Quantum Hydra (Super) | `QuantumHydra.py` | 29 KB | Option A | ✅ Ready |
| Quantum Hydra (Hybrid) | `QuantumHydraHybrid.py` | 21 KB | Option B | ✅ Ready |
| Quantum Mamba (Super) | `QuantumMamba.py` | 16 KB | Option A | ✅ Ready |
| Quantum Mamba (Hybrid) | `QuantumMambaHybrid.py` | 24 KB | Option B | ✅ Ready |
| True Classical Hydra | `TrueClassicalHydra.py` | 16 KB | Baseline | ✅ Verified |
| True Classical Mamba | `TrueClassicalMamba.py` | 11 KB | Baseline | ✅ Verified |

**Total model files: 6**

---

## 🔬 Research Questions

### Primary Questions

1. **Do quantum models outperform classical baselines?**
   - Compare Quantum Hydra (A & B) vs True Classical Hydra
   - Compare Quantum Mamba (A & B) vs True Classical Mamba

2. **Is quantum advantage architecture-specific or general?**
   - If Quantum Hydra wins but Quantum Mamba doesn't → Architecture-specific
   - If both quantum models win → General quantum advantage
   - If neither wins → No quantum advantage (yet)

3. **Does quantum superposition help?**
   - Option A (superposition) vs Option B (hybrid) for Hydra
   - Option A (superposition) vs Option B (hybrid) for Mamba

4. **Which architecture is best for quantum implementations?**
   - Quantum Hydra vs Quantum Mamba
   - May reveal which classical architectures benefit most from quantization

### Secondary Questions

5. **What's the accuracy-time trade-off?**
   - Quantum models may be slower but more accurate
   - Or faster with fewer parameters

6. **Which branches contribute most?** (Option B only)
   - Hydra: shift vs flip vs diagonal
   - Mamba: SSM vs gating vs skip

7. **How do models scale with qubits?**
   - Test with n_qubits = 4, 6, 8
   - Larger qubit counts may show quantum advantages

---

## 📋 Comparison Framework

### Current Comparison Script

**File:** `compare_all_models.py`

**Models compared:**
- ✅ Quantum Hydra (Superposition)
- ✅ Quantum Hydra (Hybrid)
- ✅ True Classical Hydra
- ✅ True Classical Mamba

**Missing:**
- ❌ Quantum Mamba (Superposition)
- ❌ Quantum Mamba (Hybrid)

### Extending the Comparison

To conduct the full ablation study, you have **two options**:

#### Option 1: Extend `compare_all_models.py`

Add Quantum Mamba models to the existing comparison script:

```python
# Add imports
from QuantumMamba import QuantumMambaTS
from QuantumMambaHybrid import QuantumMambaHybridTS

# Add model initialization
model_quantum_mamba_super = QuantumMambaTS(
    n_qubits=args.n_qubits,
    n_timesteps=n_timesteps,
    qlcu_layers=args.qlcu_layers,
    gate_layers=2,
    feature_dim=n_channels,
    output_dim=n_classes,
    dropout=0.1,
    device=device
)

model_quantum_mamba_hybrid = QuantumMambaHybridTS(
    n_qubits=args.n_qubits,
    n_timesteps=n_timesteps,
    qlcu_layers=args.qlcu_layers,
    gate_layers=2,
    feature_dim=n_channels,
    output_dim=n_classes,
    dropout=0.1,
    device=device
)

# Add to results dictionary
results['quantum_mamba_super'] = train_model(model_quantum_mamba_super, ...)
results['quantum_mamba_hybrid'] = train_model(model_quantum_mamba_hybrid, ...)
```

#### Option 2: Create Multiple Comparison Scripts

Create separate comparisons:
- `compare_quantum_hydra.py` - Hydra variants only (2 quantum + 1 classical)
- `compare_quantum_mamba.py` - Mamba variants only (2 quantum + 1 classical)
- `compare_all_6_models.py` - All models together

---

## 📈 Metrics to Track

### Performance Metrics

| Metric | Purpose |
|--------|---------|
| **Test Accuracy** | Primary performance measure |
| **Test AUC** | Robust to class imbalance |
| **Test F1 Score** | Balanced precision/recall |
| **Validation Accuracy** | Track overfitting |

### Efficiency Metrics

| Metric | Purpose |
|--------|---------|
| **Training Time** | Computational cost |
| **Model Parameters** | Model size |
| **Forward Pass Time** | Inference speed |
| **Memory Usage** | Hardware requirements |

### Quantum-Specific Metrics

| Metric | Purpose | Models |
|--------|---------|--------|
| **Branch Contributions** | Which path dominates | Option B only |
| **Complex Coefficients** | α, β, γ values | Option A only |
| **Quantum Circuit Depth** | Circuit complexity | All quantum |
| **Number of Measurements** | Quantum overhead | All quantum |

---

## 🧪 Experimental Design

### Dataset

**PhysioNet EEG Motor Imagery**
- Task: Left/right hand movement classification
- Channels: 64
- Sampling rate: 160 Hz → 200 timesteps
- Classes: 2 (binary classification)

### Hyperparameters

#### Fixed Across All Models
```python
seed = 2024
batch_size = 16
learning_rate = 1e-3
n_epochs = 50
optimizer = Adam
scheduler = ReduceLROnPlateau
early_stopping = True (patience=10)
```

#### Model-Specific

**Quantum Models:**
```python
n_qubits = 6  # Test: 4, 6, 8
qlcu_layers = 2
gate_layers = 2  # Mamba only
dropout = 0.1
```

**Classical Models:**
```python
# Hydra
hidden_dim = 64  # Match quantum model capacity

# Mamba
d_model = 128
d_state = 16
```

### Training Protocol

1. **Data Split:**
   - Train: 70%
   - Validation: 15%
   - Test: 15%
   - Fixed seed for reproducibility

2. **Training:**
   - Cross-entropy loss
   - Adam optimizer
   - ReduceLROnPlateau (factor=0.5, patience=5)
   - Early stopping (patience=10)

3. **Evaluation:**
   - Best validation model used for testing
   - Report mean ± std over 3 seeds

---

## 📊 Expected Comparison Matrix

### Accuracy Comparison (Hypothetical)

| Model | Test Acc | Test AUC | Test F1 | Time (s) | Params |
|-------|----------|----------|---------|----------|--------|
| **Quantum Models** |
| Q-Hydra (Super) | ? | ? | ? | ? | ~2k |
| Q-Hydra (Hybrid) | ? | ? | ? | ? | ~2k |
| Q-Mamba (Super) | ? | ? | ? | ? | ~1.5k |
| Q-Mamba (Hybrid) | ? | ? | ? | ? | ~1.5k |
| **Classical Baselines** |
| Classical Hydra | ? | ? | ? | ? | ~5k |
| Classical Mamba | ? | ? | ? | ? | ~8k |

Fill in after experiments!

### Possible Outcomes

**Scenario 1: General Quantum Advantage ✨**
```
Q-Hydra (Super) > Classical Hydra
Q-Mamba (Super) > Classical Mamba
→ Quantum superposition helps across architectures
```

**Scenario 2: Architecture-Specific**
```
Q-Hydra (Super) > Classical Hydra
Q-Mamba (Super) ≈ Classical Mamba
→ Quantum advantages specific to Hydra structure
```

**Scenario 3: Hybrid Works Better**
```
Q-Hydra (Hybrid) > Q-Hydra (Super)
Q-Mamba (Hybrid) > Q-Mamba (Super)
→ Classical combination more stable than interference
```

**Scenario 4: No Quantum Advantage (Yet)**
```
All quantum ≈ Classical
→ Need more qubits, better encoding, or different tasks
```

---

## 🎨 Visualization Plan

### Figure 1: Training Curves (3×3 grid)
- Row 1: Loss curves (train, val, test)
- Row 2: Accuracy curves
- Row 3: Learning rate schedule

### Figure 2: Final Performance (2×3 grid)
- Test accuracy bars
- Test AUC bars
- Test F1 bars
- Training time bars
- Parameter count bars
- Accuracy vs Time scatter

### Figure 3: Quantum Analysis (Option A)
- Complex coefficient trajectories (α, β, γ)
- Coefficient magnitudes over training
- Phase evolution

### Figure 4: Branch Analysis (Option B)
- Branch contribution bars (w₁, w₂, w₃)
- Contribution evolution over training
- Per-sample contribution distributions

---

## 🚀 Running Experiments

### Quick Test (Mini Dataset)

```bash
# Test Quantum Hydra variants
python compare_all_models.py \
    --n-qubits=4 \
    --qlcu-layers=1 \
    --n-epochs=10 \
    --sample-size=5 \
    --batch-size=8

# Test individual models
python QuantumMamba.py
python QuantumMambaHybrid.py
```

### Full Experiment

```bash
# Submit SLURM job
sbatch run_all_models_comparison.sh

# Or run locally (CPU/GPU auto-detect)
python compare_all_models.py \
    --n-qubits=6 \
    --qlcu-layers=2 \
    --hidden-dim=64 \
    --d-model=128 \
    --d-state=16 \
    --n-epochs=50 \
    --batch-size=16 \
    --lr=1e-3 \
    --sample-size=10 \
    --sampling-freq=100 \
    --seed=2024 \
    --device=cuda
```

### Ablation Studies

**Study 1: Qubit Scaling**
```bash
for n_qubits in 4 6 8; do
    python compare_all_models.py --n-qubits=$n_qubits --seed=2024
done
```

**Study 2: Circuit Depth**
```bash
for layers in 1 2 3; do
    python compare_all_models.py --qlcu-layers=$layers --seed=2024
done
```

**Study 3: Multi-Seed Robustness**
```bash
for seed in 2024 2025 2026; do
    python compare_all_models.py --seed=$seed
done
```

---

## 📁 Output Files

### Results Files

After running experiments, you'll get:

```
all_models_comparison_YYYYMMDD_HHMMSS.json  # Raw results
all_models_comparison_YYYYMMDD_HHMMSS.pdf   # Visualization
logs/compare_all_models_<job_id>.out        # Console log
logs/compare_all_models_<job_id>.err        # Error log
```

### JSON Structure

```json
{
  "quantum_super": {
    "model_name": "Quantum Hydra (Superposition)",
    "test_acc": 0.85,
    "test_auc": 0.87,
    "test_f1": 0.84,
    "training_time": 1200.5,
    "n_params": 2048,
    "history": {...}
  },
  "quantum_hybrid": {...},
  "hydra": {...},
  "mamba": {...}
}
```

---

## 🔍 Analysis Workflow

### Step 1: Run Experiments

```bash
# Option 1: All models at once
python compare_all_models.py --seed=2024

# Option 2: Separate comparisons
python compare_quantum_hydra.py --seed=2024
python compare_quantum_mamba.py --seed=2024
```

### Step 2: Extract Results

```bash
# View summary
cat all_models_comparison_*.json | jq '.[] | {name: .model_name, acc: .test_acc}'

# Extract to CSV
python -c "
import json
with open('all_models_comparison_*.json') as f:
    data = json.load(f)
    for model, results in data.items():
        print(f\"{results['model_name']},{results['test_acc']},{results['training_time']}\")
" > results_summary.csv
```

### Step 3: Statistical Analysis

```python
import numpy as np
from scipy import stats

# Load results from multiple seeds
results_seed1 = load_json('results_seed_2024.json')
results_seed2 = load_json('results_seed_2025.json')
results_seed3 = load_json('results_seed_2026.json')

# Compute mean ± std
quantum_accs = [r['quantum_super']['test_acc'] for r in [results_seed1, ...]]
classical_accs = [r['hydra']['test_acc'] for r in [results_seed1, ...]]

print(f"Quantum: {np.mean(quantum_accs):.3f} ± {np.std(quantum_accs):.3f}")
print(f"Classical: {np.mean(classical_accs):.3f} ± {np.std(classical_accs):.3f}")

# Statistical significance
t_stat, p_value = stats.ttest_ind(quantum_accs, classical_accs)
print(f"p-value: {p_value:.4f}")
```

### Step 4: Create Publication Figures

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot with error bars
models = ['Q-Hydra\n(Super)', 'Q-Hydra\n(Hybrid)', 'Q-Mamba\n(Super)',
          'Q-Mamba\n(Hybrid)', 'Classical\nHydra', 'Classical\nMamba']
accs = [...]  # Fill from results
errors = [...]  # Standard deviations

plt.figure(figsize=(10, 6))
plt.bar(models, accs, yerr=errors, capsize=5)
plt.ylabel('Test Accuracy')
plt.title('Quantum vs Classical State-Space Models')
plt.ylim([0.5, 1.0])
plt.savefig('ablation_accuracy.pdf', bbox_inches='tight', dpi=300)
```

---

## 📚 Documentation Files

### Model Documentation

- **`QUANTUM_HYDRA_README.md`** - Quantum Hydra detailed guide
- **`QUANTUM_MAMBA_README.md`** - Quantum Mamba detailed guide
- **`COMPARISON_README.md`** - Original 4-model comparison
- **`ABLATION_STUDY_GUIDE.md`** - This file

### Code Files

- **Models:**
  - `QuantumHydra.py`, `QuantumHydraHybrid.py`
  - `QuantumMamba.py`, `QuantumMambaHybrid.py`
  - `TrueClassicalHydra.py`, `TrueClassicalMamba.py`

- **Comparison:**
  - `compare_all_models.py` - Current 4-model comparison
  - `run_all_models_comparison.sh` - SLURM script

- **Data:**
  - `Load_PhysioNet_EEG.py` - EEG data loader

---

## ✅ Pre-Flight Checklist

Before running full experiments:

- [ ] All 6 model files exist and tested
- [ ] Data loader works (`Load_PhysioNet_EEG.py`)
- [ ] GPU available (`nvidia-smi`)
- [ ] Conda environment activated (`qml_eeg`)
- [ ] Hyperparameters decided
- [ ] Seeds fixed for reproducibility
- [ ] Output directory exists (`mkdir -p logs`)
- [ ] Enough disk space for checkpoints
- [ ] Enough GPU memory (16GB+ recommended)
- [ ] Time allocation sufficient (12+ hours for full run)

---

## 🎯 Publication Checklist

Results ready for publication when:

- [ ] All 6 models trained
- [ ] Multiple seeds tested (≥3)
- [ ] Statistical significance computed
- [ ] Publication-quality figures created
- [ ] Results table formatted
- [ ] Branch analysis completed (Option B)
- [ ] Training curves saved
- [ ] Model checkpoints backed up
- [ ] Hyperparameters documented
- [ ] Ablation studies completed (qubits, depth, seeds)

---

## 💡 Tips for Success

### 1. Start Small

```bash
# Test with mini dataset first
python compare_all_models.py --sample-size=5 --n-epochs=10 --n-qubits=4
```

### 2. Use Checkpointing

```bash
# Enable resume capability
python compare_all_models.py --resume --job-id=exp001
```

### 3. Monitor Training

```bash
# Watch logs in real-time
tail -f logs/compare_all_models_*.out
```

### 4. Save Results

```bash
# Backup results immediately
cp all_models_comparison_*.json results_backup/
cp all_models_comparison_*.pdf figures_backup/
```

### 5. Document Everything

Keep a lab notebook with:
- Hyperparameters used
- Observations during training
- Unexpected behaviors
- Ideas for improvements

---

## 🔮 Future Directions

### After Initial Ablation Study

1. **Noise Robustness**
   - Add quantum noise models
   - Test with realistic error rates

2. **Larger Datasets**
   - Scale to more EEG subjects
   - Try fMRI data
   - Test on other time-series tasks

3. **Architectural Variants**
   - Deeper quantum circuits
   - More qubits (8, 12, 16)
   - Different ansatz designs

4. **Hybrid Approaches**
   - Combine Option A and B
   - Ensemble of quantum models
   - Classical-quantum ensembles

5. **Theoretical Analysis**
   - Expressivity bounds
   - Entanglement entropy
   - Quantum advantage proofs

---

## 📞 Support

If you encounter issues:

1. Check model test outputs: `python QuantumMamba.py`
2. Verify data loading: `python Load_PhysioNet_EEG.py`
3. Check GPU: `nvidia-smi`
4. Review logs: `tail logs/*.err`

---

**Author:** Junghoon Park
**Last Updated:** October 2024
**Status:** Ready for experiments
**Purpose:** Complete ablation study of quantum vs classical SSMs

---

## 🎉 Summary

You now have:
- ✅ **6 models** ready (4 quantum + 2 classical)
- ✅ **Both tested** and working
- ✅ **Comprehensive documentation** for each
- ✅ **Comparison framework** in place
- ✅ **Clear research questions** defined
- ✅ **Experimental protocol** documented

**Next step:** Run the experiments and discover if quantum advantages exist! 🚀
