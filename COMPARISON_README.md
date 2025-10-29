# Comprehensive Model Comparison: Quantum vs Classical State-Space Models

This document describes the updated comparison framework that compares **two quantum Hydra models** against **two corrected classical baselines**.

## 🎯 What Changed

### Previous Comparison (`compare_quantum_hydra.py`)
- ❌ Used `ClassicalHydra.py` - flawed baseline implementation
- ⚠️ Only compared against one classical model
- ⚠️ Did not include Mamba baseline

### **New Comparison (`compare_all_models.py`)**
- ✅ Uses `TrueClassicalHydra.py` - faithful implementation of Hwang et al. (2024)
- ✅ Uses `TrueClassicalMamba.py` - faithful implementation of Gu & Dao (2024)
- ✅ Compares **4 models** total for comprehensive analysis
- ✅ More rigorous comparison with corrected baselines

---

## 📊 Models Being Compared

### **1. Quantum Hydra (Superposition) - Option A** 🔵

**File:** `QuantumHydra.py`

**Mathematical Formulation:**
```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
where α, β, γ ∈ ℂ (complex coefficients)
```

**Key Features:**
- Quantum superposition of three branches
- Complex-valued trainable coefficients
- Single measurement on combined quantum state
- Potential for quantum interference effects
- Fewer quantum circuit calls per forward pass

**Advantages:**
- ✓ Quantum interference may capture non-classical correlations
- ✓ Exponential state space (2^n)
- ✓ Fewer measurements required

**Limitations:**
- ✗ Different semantics from classical Hydra
- ✗ Sensitive to quantum decoherence
- ✗ Complex coefficient optimization

---

### **2. Quantum Hydra (Hybrid) - Option B** 🟢

**File:** `QuantumHydraHybrid.py`

**Mathematical Formulation:**
```
y = w₁·y₁ + w₂·y₂ + w₃·y₃
where w₁, w₂, w₃ ∈ ℝ (real weights)
```

**Key Features:**
- Three independent quantum circuits
- Classical weighted combination of measurements
- Real-valued trainable weights
- Faithful to classical Hydra's addition semantics
- More interpretable branch contributions

**Advantages:**
- ✓ Preserves classical Hydra semantics
- ✓ Interpretable (can analyze each branch)
- ✓ More robust to quantum noise
- ✓ Real-valued weights (easier optimization)

**Limitations:**
- ✗ No quantum interference
- ✗ More quantum circuit calls
- ✗ Higher computational cost

---

### **3. True Classical Hydra - Corrected Baseline** 🔴

**File:** `TrueClassicalHydra.py`

**Mathematical Formulation:**
```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
Y = W · QS(X)
```

**Key Features:**
- **Faithful implementation** of Hwang et al. (2024) paper
- Unidirectional processing (forward only)
- Semi-separable matrix operations via depthwise convolutions
- Three-branch architecture with shift and flip operations
- Corrected from flawed `ClassicalHydra.py`

**Why this is important:**
- Previous `ClassicalHydra.py` had incorrect implementation
- This version matches the actual Hydra paper
- Provides fair comparison for quantum models

**Architecture:**
- Semi-separable via 1D conv (kernel_size=3, groups=d_model)
- Circular shift operation
- Sequence flip (reversal)
- Diagonal skip connection
- LayerNorm and residual connections

---

### **4. True Classical Mamba - Additional Baseline** 🟣

**File:** `TrueClassicalMamba.py`

**Mathematical Formulation:**
```
Selective SSM:
  x[t] = A * x[t-1] + B(u) * u[t]
  y[t] = C(u) * x[t] + D * u[t]

where B, C, dt are input-dependent
```

**Key Features:**
- **Faithful implementation** of Gu & Dao (2024) paper
- Selective state-space model (input-dependent B, C, dt)
- Unidirectional processing via recurrent scan
- RMSNorm instead of LayerNorm
- Gated MLP block structure

**Why include Mamba:**
- State-of-the-art classical SSM baseline
- Recently surpassed Transformers on many tasks
- Provides additional comparison point
- Tests if quantum models can beat modern classical SSMs

**Architecture:**
- Selective SSM core with input-dependent parameters
- 1D depthwise convolution for local context
- SiLU gating mechanism
- Efficient unidirectional scan

---

## 🔬 Comparison Framework

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Test Accuracy** | Final classification accuracy on held-out test set |
| **Test AUC** | Area under ROC curve (binary classification) |
| **Test F1 Score** | Weighted F1 score for balanced evaluation |
| **Training Time** | Wall-clock time for complete training |
| **Model Parameters** | Total trainable parameters |
| **Best Val Acc** | Highest validation accuracy during training |

### Evaluation Protocol

1. **Dataset:** PhysioNet EEG Motor Imagery (left/right hand)
2. **Split:** Train/Val/Test with fixed random seed
3. **Training:**
   - Cross-entropy loss
   - Adam optimizer
   - ReduceLROnPlateau scheduler
   - Early stopping based on validation accuracy
4. **Reproducibility:** Fixed seed (2024) across all models

---

## 🚀 How to Run

### Quick Start

```bash
# Run comparison with default settings
python compare_all_models.py
```

### Custom Configuration

```bash
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

### SLURM Submission

```bash
# Make executable
chmod +x run_all_models_comparison.sh

# Submit to queue
sbatch run_all_models_comparison.sh

# Check status
squeue -u $USER

# View output
tail -f logs/compare_all_models_<job_id>.out
```

---

## 📈 Output Files

After running the comparison, you'll get:

### 1. **JSON Results**
**File:** `all_models_comparison_YYYYMMDD_HHMMSS.json`

Contains:
- Training history (loss, accuracy per epoch)
- Final test metrics (accuracy, AUC, F1)
- Training time
- Model parameters
- Confusion matrices

**Example structure:**
```json
{
  "quantum_super": {
    "model_name": "Quantum Hydra (Superposition)",
    "test_acc": 0.8523,
    "test_auc": 0.8745,
    "test_f1": 0.8501,
    "training_time": 1245.32,
    "n_params": 2048,
    "history": {...}
  },
  "quantum_hybrid": {...},
  "hydra": {...},
  "mamba": {...}
}
```

### 2. **Comparison Plot**
**File:** `all_models_comparison_YYYYMMDD_HHMMSS.pdf`

**9-panel figure showing:**
1. Training loss curves
2. Validation loss curves
3. Validation accuracy curves
4. Test accuracy bar chart
5. Test AUC bar chart
6. Test F1 score bar chart
7. Training time bar chart
8. Model parameters comparison
9. Accuracy vs Time scatter plot

### 3. **Log File**
**File:** `logs/comparison_<job_id>.log`

Complete console output with:
- Epoch-by-epoch training progress
- Validation metrics
- Final test results
- Timing information

---

## 📊 Expected Results

### Hypothesis

**Best Performance:**
- **Mamba** likely highest accuracy (state-of-the-art classical SSM)
- **Hydra** comparable to Mamba
- **Quantum models** may match or slightly outperform on small datasets

**Training Speed:**
- **Hydra/Mamba** fastest (no quantum overhead)
- **Quantum Superposition** moderate (fewer circuit calls)
- **Quantum Hybrid** slowest (three separate circuits)

**Model Size:**
- **Quantum models** fewer parameters (quantum encoding efficiency)
- **Classical models** more parameters

### Research Questions

1. **Can quantum models match classical baselines?**
   - Compare Quantum Hydra vs True Classical Hydra

2. **Does quantum superposition help?**
   - Compare Option A vs Option B

3. **How do quantum models compare to SOTA?**
   - Compare Quantum models vs Mamba

4. **What's the accuracy-time trade-off?**
   - Analyze scatter plot in panel 9

---

## 🔍 Detailed Comparison Table

| Aspect | Quantum Super | Quantum Hybrid | True Hydra | True Mamba |
|--------|--------------|----------------|------------|------------|
| **File** | `QuantumHydra.py` | `QuantumHydraHybrid.py` | `TrueClassicalHydra.py` | `TrueClassicalMamba.py` |
| **Branch Combination** | Quantum superposition | Classical addition | Classical addition | SSM recurrence |
| **Coefficients** | Complex (α, β, γ) | Real (w₁, w₂, w₃) | Fixed structure | Input-dependent |
| **Processing** | Unidirectional | Unidirectional | Bidirectional | Unidirectional |
| **Quantum Circuits** | Yes | Yes | No | No |
| **Interpretability** | Low | High | High | Moderate |
| **Paper** | N/A (novel) | N/A (novel) | Hwang et al. 2024 | Gu & Dao 2024 |
| **Quantum Advantage** | Possible | Limited | N/A | N/A |

---

## 🛠️ Troubleshooting

### Issue: Import Errors

```bash
# Check all required files exist
ls QuantumHydra.py
ls QuantumHydraHybrid.py
ls TrueClassicalHydra.py
ls TrueClassicalMamba.py
ls Load_PhysioNet_EEG_NoPrompt.py
```

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python compare_all_models.py --batch-size=8

# Or use CPU
python compare_all_models.py --device=cpu
```

### Issue: Slow Training

```bash
# Reduce sample size
python compare_all_models.py --sample-size=5

# Reduce epochs
python compare_all_models.py --n-epochs=25
```

---

## 📚 References

### Papers

1. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - https://arxiv.org/pdf/2407.09941
   - Classical Hydra architecture

2. **Gu & Dao (2024)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/html/2312.00752v2
   - Mamba SSM architecture

3. **Goldberger et al. (2000)** - "PhysioBank, PhysioToolkit, and PhysioNet"
   - PhysioNet EEG dataset

### Code Files

- `QUANTUM_HYDRA_README.md` - Detailed documentation of quantum models
- `compare_quantum_hydra.py` - Old comparison (deprecated)
- `ClassicalHydra.py` - Old flawed baseline (deprecated)

---

## 💡 Key Takeaways

### Why This Comparison Matters

1. **Rigorous Baselines:** Using faithful classical implementations ensures fair comparison
2. **Multiple Baselines:** Comparing against both Hydra and Mamba tests quantum models thoroughly
3. **Comprehensive Metrics:** Multiple metrics (accuracy, AUC, F1, time) give full picture
4. **Reproducible:** Fixed seeds and detailed logging enable reproduction

### What Makes This Better

| Aspect | Old Comparison | New Comparison |
|--------|----------------|----------------|
| **Classical Hydra** | Flawed implementation | Faithful to paper ✓ |
| **Mamba Baseline** | Not included | Included ✓ |
| **Number of Models** | 3 | 4 ✓ |
| **Metrics** | Basic | Comprehensive ✓ |
| **Visualization** | 6 panels | 9 panels ✓ |

---

## 🎯 Next Steps

After running the comparison:

1. **Analyze Results:**
   ```bash
   # View results
   cat all_models_comparison_*.json | jq '.[] | {name: .model_name, acc: .test_acc}'

   # Open plots
   open all_models_comparison_*.pdf
   ```

2. **Tune Hyperparameters:**
   - Adjust `--n-qubits` (4, 6, 8)
   - Try different `--qlcu-layers` (1, 2, 3)
   - Experiment with `--d-model` (64, 128, 256)

3. **Extended Analysis:**
   - Run with multiple seeds for statistical significance
   - Test on larger `--sample-size`
   - Try different EEG tasks

4. **Document Findings:**
   - Update `QUANTUM_HYDRA_RESULTS.md`
   - Create visualizations
   - Write paper/report

---

## ✅ Checklist for Running

- [ ] Activate conda environment: `conda activate ./conda-envs/qml_env`
- [ ] Verify all model files exist
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Create logs directory: `mkdir -p logs`
- [ ] Set desired hyperparameters
- [ ] Run comparison: `python compare_all_models.py` or `sbatch run_all_models_comparison.sh`
- [ ] Monitor progress: `tail -f logs/compare_all_models_*.out`
- [ ] Analyze results after completion

---

**Author:** Junghoon Park
**Last Updated:** October 2024
**Status:** Production-ready
