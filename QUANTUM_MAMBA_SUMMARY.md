# Quantum Mamba Implementation - Summary

## ✅ What Was Created

### New Model Files (2 files)

1. **`QuantumMamba.py` (16 KB)**
   - Quantum Mamba with **superposition** (Option A)
   - Complex coefficients: α, β, γ ∈ ℂ
   - Quantum interference between SSM, gating, and skip paths
   - Status: ✅ Tested and working

2. **`QuantumMambaHybrid.py` (24 KB)**
   - Quantum Mamba with **classical combination** (Option B)
   - Real weights: w₁, w₂, w₃ ∈ ℝ
   - Independent quantum circuits, classical addition
   - Status: ✅ Tested and working

### Documentation Files (2 files)

3. **`QUANTUM_MAMBA_README.md` (17 KB)**
   - Comprehensive documentation for Quantum Mamba models
   - Architecture details
   - Usage examples
   - Comparison with Quantum Hydra

4. **`ABLATION_STUDY_GUIDE.md` (16 KB)**
   - Complete guide for ablation studies
   - 6-model comparison framework
   - Research questions
   - Experimental protocol

---

## 📊 Complete Model Set (6 Models)

You now have all models ready for ablation studies:

### Quantum Models (4 total)
1. ✅ Quantum Hydra (Superposition) - `QuantumHydra.py`
2. ✅ Quantum Hydra (Hybrid) - `QuantumHydraHybrid.py`
3. ✅ **Quantum Mamba (Superposition)** - `QuantumMamba.py` [NEW]
4. ✅ **Quantum Mamba (Hybrid)** - `QuantumMambaHybrid.py` [NEW]

### Classical Baselines (2 total)
5. ✅ True Classical Hydra - `TrueClassicalHydra.py`
6. ✅ True Classical Mamba - `TrueClassicalMamba.py`

---

## 🔬 Key Features

### Quantum Mamba Architecture

Both Quantum Mamba models implement three branches:

**Branch 1: Selective SSM**
- Input-dependent B, C, dt parameters (like Mamba paper)
- QLCU for state transformation
- Quantum implementation of `h[t] = A*h[t-1] + B*u[t]`

**Branch 2: Gating**
- Quantum gating mechanism
- Controlled rotations for multiplicative gating
- Mimics SiLU gating in classical Mamba

**Branch 3: Skip Connection**
- Diagonal operations (D matrix)
- Direct input-to-output path
- Ensures gradient flow

### Differences Between Options

**Option A (Superposition):**
```python
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
y = Measure(|ψ⟩)
```
- Quantum interference before measurement
- Complex coefficients
- Fewer quantum circuit calls

**Option B (Hybrid):**
```python
y₁ = Measure(|ψ_ssm⟩)
y₂ = Measure(|ψ_gate⟩)
y₃ = Measure(|ψ_skip⟩)
y = w₁·y₁ + w₂·y₂ + w₃·y₃
```
- Classical combination after measurement
- Real weights
- Easier to interpret

---

## 🎯 Research Questions

Your ablation study can now answer:

1. **Do quantum models outperform classical?**
   - Compare quantum vs classical baselines

2. **Is quantum advantage architecture-specific?**
   - Quantum Hydra vs Quantum Mamba performance

3. **Does quantum superposition help?**
   - Option A vs Option B for both architectures

4. **Which design is most practical?**
   - Accuracy, speed, interpretability trade-offs

---

## 🚀 Quick Start

### Test Individual Models

```bash
# Test Quantum Mamba (Superposition)
python QuantumMamba.py

# Test Quantum Mamba (Hybrid)
python QuantumMambaHybrid.py
```

### Run Comparison

```bash
# Current comparison (4 models)
python compare_all_models.py --n-qubits=6 --n-epochs=50

# For 6 models, extend compare_all_models.py to include:
# - QuantumMamba.py (Option A)
# - QuantumMambaHybrid.py (Option B)
```

### SLURM Submission

```bash
# Submit comparison job
sbatch run_all_models_comparison.sh
```

---

## 📚 Documentation

| File | Size | Purpose |
|------|------|---------|
| `QUANTUM_MAMBA_README.md` | 17 KB | Detailed Quantum Mamba docs |
| `QUANTUM_HYDRA_README.md` | 22 KB | Detailed Quantum Hydra docs |
| `ABLATION_STUDY_GUIDE.md` | 16 KB | Complete ablation study guide |
| `COMPARISON_README.md` | 12 KB | 4-model comparison guide |

---

## 🔍 Code Verification

Both models tested successfully:

**QuantumMamba.py output:**
```
[1] Testing QuantumMambaLayer...
  Input shape: torch.Size([4, 64])
  Output shape: torch.Size([4, 2])

[2] Testing QuantumMambaTS (full model)...
  Input shape: torch.Size([4, 64, 160])
  Output shape: torch.Size([4, 2])

[3] Checking trainable parameters...
  Total parameters: 1,564

[4] Complex superposition coefficients:
  α = (0.476+0j)
  β = (0.882+0j)
  γ = (0.704+0j)
```

**QuantumMambaHybrid.py output:**
```
[1] Testing QuantumMambaHybridLayer...
  Input shape: torch.Size([8, 10])
  Output shape: torch.Size([8, 2])

[2] Testing branch contribution analysis...
  Branch 1 (SSM) shape: torch.Size([2, 12])
  Branch 2 (Gate) shape: torch.Size([2, 12])
  Branch 3 (Skip) shape: torch.Size([2, 12])
  Branch weights: w1_ssm=0.333, w2_gate=0.333, w3_skip=0.333

[3] Testing QuantumMambaHybridTS (time-series)...
  Input shape: torch.Size([8, 64, 20])
  Output shape: torch.Size([8, 1])
```

---

## 📈 Next Steps

1. **Extend comparison script** to include all 6 models:
   ```python
   from QuantumMamba import QuantumMambaTS
   from QuantumMambaHybrid import QuantumMambaHybridTS
   ```

2. **Run experiments** on PhysioNet EEG dataset

3. **Analyze results**:
   - Accuracy comparison
   - Training time
   - Parameter efficiency
   - Branch contributions (Option B)

4. **Create visualizations**:
   - Performance bars
   - Training curves
   - Accuracy vs time scatter

---

## 🎉 Summary

**Created:**
- ✅ 2 new quantum model files
- ✅ 2 comprehensive documentation files
- ✅ Complete ablation study framework

**Status:**
- ✅ All models tested and working
- ✅ Ready for experiments
- ✅ Documentation complete

**Ready for:**
- 🚀 Ablation studies
- 🚀 6-model comparison
- 🚀 Publication-quality results

---

**Author:** Junghoon Park
**Date:** October 2024
**Purpose:** Ablation study of quantum vs classical SSMs
