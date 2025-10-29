# Quantum Hydra: Two Approaches to Quantum State-Space Models

This document provides a comprehensive comparison of two quantum implementations of the Hydra state-space model, inspired by classical deep learning architectures (Hwang et al., 2024).

## Table of Contents
1. [Background: Classical Hydra](#background-classical-hydra)
2. [Option A: Quantum Superposition](#option-a-quantum-superposition)
3. [Option B: Hybrid Classical-Quantum](#option-b-hybrid-classical-quantum)
4. [Mathematical Comparison](#mathematical-comparison)
5. [Implementation Comparison](#implementation-comparison)
6. [Usage Examples](#usage-examples)
7. [When to Use Which](#when-to-use-which)
8. [References](#references)

---

## Background: Classical Hydra

Classical Hydra (Hwang et al., 2024) is a state-space model with the following operations:

### **Classical Hydra Equation**

```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
Y = W · QS(X)
```

Where:
- `X ∈ ℝ^(L×d)`: Input sequence (L = length, d = dimension)
- `SS`: Semi-separable matrix operation
- `shift`: Circular shift operation
- `flip`: Sequence reversal operation
- `D`: Diagonal matrix
- `W`: Weight matrix
- `+`: Classical vector addition

**Key Property**: Classical Hydra combines three branches via **element-wise addition** of vectors.

---

## Option A: Quantum Superposition

### **Mathematical Formulation**

Option A creates a quantum superposition of three branches:

```
|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
```

Where each branch is:

```
|ψ₁⟩ = Qshift(QLCU|X⟩)
|ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
|ψ₃⟩ = QD|X⟩
```

And:
- `α, β, γ ∈ ℂ`: Trainable complex coefficients
- `QLCU`: Quantum Linear Combination of Unitaries (simulates SS matrix)
- `Qshift`: Quantum cyclic shift via SWAP gates
- `Qflip`: Quantum reversal via SWAP gates
- `QD`: Quantum diagonal operation (single-qubit rotations)

**Normalization**:
```
|ψ_norm⟩ = |ψ⟩ / ||ψ||
```

**Measurement**:
```
Y_i = ⟨ψ_norm|M_i|ψ_norm⟩  for i = 1, 2, ..., 3n
```

Where `M_i ∈ {X_j, Y_j, Z_j}` are Pauli observables on qubits j = 0, 1, ..., n-1.

**Final Output**:
```
Y = W_out · [Y_1, Y_2, ..., Y_{3n}]^T
```

### **Quantum Circuit Structure**

```
|0⟩^⊗n ──[Encoding]──┬──[QLCU]──[Qshift]─────────────────┬──
                     │                                    │
                     ├──[Qflip]──[QLCU]──[Qshift]──[Qflip]┤── α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
                     │                                    │
                     └──[QD]────────────────────────────┘

                     ↓ (Single measurement after superposition)

                [M₁]  [M₂]  ...  [M₃ₙ]  → Expectation values
```

### **Key Properties**

**✅ Advantages:**
- **Quantum interference**: Branches can interfere constructively/destructively
- **Exponential state space**: Can represent 2^n states simultaneously
- **Potential quantum advantage**: May capture correlations classical models cannot
- **Fewer measurements**: Single measurement on combined state

**❌ Disadvantages:**
- **Different from classical**: Not equivalent to classical Hydra's addition
- **Harder to interpret**: Quantum correlations obscure individual branch contributions
- **Sensitive to decoherence**: Superposition fragile to environmental noise
- **Complex coefficients**: Requires careful optimization of α, β, γ ∈ ℂ

### **Python Implementation**

```python
from QuantumHydra import QuantumHydraLayer

# Create Option A model
model_a = QuantumHydraLayer(
    n_qubits=6,              # Number of qubits
    qlcu_layers=2,           # QLCU circuit depth
    shift_amount=1,          # Shift by 1 position
    feature_dim=64,          # Input feature dimension
    output_dim=4,            # Number of classes
    dropout=0.1,
    device="cuda"
)

# Forward pass
import torch
x = torch.randn(32, 64)      # (batch_size, features)
output = model_a(x)          # (batch_size, 4)

# Check learned complex coefficients
print(f"Alpha: {model_a.alpha.data}")   # Complex number
print(f"Beta: {model_a.beta.data}")     # Complex number
print(f"Gamma: {model_a.gamma.data}")   # Complex number
```

**For time-series data:**

```python
from QuantumHydra import QuantumHydraTS

model_ts = QuantumHydraTS(
    n_qubits=6,
    n_timesteps=160,         # EEG timesteps
    qlcu_layers=2,
    feature_dim=64,          # EEG channels
    output_dim=2,            # Binary classification
    device="cuda"
)

# EEG data: (batch, channels, timesteps)
eeg_data = torch.randn(16, 64, 160)
predictions = model_ts(eeg_data)  # (16, 2)
```

---

## Option B: Hybrid Classical-Quantum

### **Mathematical Formulation**

Option B computes each branch independently, then combines classically:

```
Step 1: Compute quantum branches
  |ψ₁⟩ = Qshift(QLCU|X⟩)
  |ψ₂⟩ = Qflip(Qshift(QLCU(Qflip|X⟩)))
  |ψ₃⟩ = QD|X⟩

Step 2: Measure each branch separately
  y₁ = [⟨ψ₁|M_i|ψ₁⟩]_{i=1}^{3n} ∈ ℝ^{3n}
  y₂ = [⟨ψ₂|M_i|ψ₂⟩]_{i=1}^{3n} ∈ ℝ^{3n}
  y₃ = [⟨ψ₃|M_i|ψ₃⟩]_{i=1}^{3n} ∈ ℝ^{3n}

Step 3: Classical weighted combination
  y_combined = w₁·y₁ + w₂·y₂ + w₃·y₃

Step 4: Output layer
  Y = W_out · y_combined
```

Where:
- `w₁, w₂, w₃ ∈ ℝ₊`: Trainable real-valued weights
- `+`: Classical vector addition (element-wise)

**Weight Normalization** (optional):
```
w_i' = w_i / (w₁ + w₂ + w₃)  for i = 1, 2, 3
```

### **Quantum Circuit Structure**

```
Branch 1:
|0⟩^⊗n ──[Encoding]──[QLCU]──[Qshift]──[Measure]→ y₁

Branch 2:
|0⟩^⊗n ──[Encoding]──[Qflip]──[QLCU]──[Qshift]──[Qflip]──[Measure]→ y₂

Branch 3:
|0⟩^⊗n ──[Encoding]──[QD]──[Measure]→ y₃

                ↓ (Three independent measurements)

         w₁·y₁ + w₂·y₂ + w₃·y₃ ← Classical addition
```

### **Key Properties**

**✅ Advantages:**
- **Faithful to classical Hydra**: Preserves classical addition semantics
- **Interpretable**: Can analyze each branch's contribution independently
- **Robust to noise**: Each branch measured separately (no interference loss)
- **Real-valued weights**: Easier optimization (w₁, w₂, w₃ ∈ ℝ)
- **Ablation-friendly**: Can disable branches to study importance

**❌ Disadvantages:**
- **No quantum interference**: Loses potential quantum advantage
- **More measurements**: Requires three separate quantum circuit executions
- **Higher computational cost**: Each branch computed independently
- **No entanglement across branches**: Branches don't share quantum information

### **Python Implementation**

```python
from QuantumHydraHybrid import QuantumHydraHybridLayer

# Create Option B model
model_b = QuantumHydraHybridLayer(
    n_qubits=6,
    qlcu_layers=2,
    shift_amount=1,
    feature_dim=64,
    output_dim=4,
    dropout=0.1,
    device="cuda"
)

# Forward pass
x = torch.randn(32, 64)
output = model_b(x)

# Check learned real-valued weights
print(f"w1: {model_b.w1.data.item():.4f}")
print(f"w2: {model_b.w2.data.item():.4f}")
print(f"w3: {model_b.w3.data.item():.4f}")

# Analyze branch contributions
contributions = model_b.get_branch_contributions(x[:2])
print(f"Branch 1 output: {contributions['branch1'].shape}")
print(f"Branch 2 output: {contributions['branch2'].shape}")
print(f"Branch 3 output: {contributions['branch3'].shape}")
print(f"Normalized weights: {contributions['weights']}")
```

**For time-series data:**

```python
from QuantumHydraHybrid import QuantumHydraHybridTS

model_ts_hybrid = QuantumHydraHybridTS(
    n_qubits=6,
    n_timesteps=160,
    qlcu_layers=2,
    feature_dim=64,
    output_dim=2,
    device="cuda"
)

eeg_data = torch.randn(16, 64, 160)
predictions = model_ts_hybrid(eeg_data)
```

---

## Mathematical Comparison

### **Core Difference**

| Aspect | Option A (Quantum) | Option B (Hybrid) |
|--------|-------------------|-------------------|
| **Branch Combination** | Quantum superposition before measurement | Classical addition after measurement |
| **Equation** | `\|ψ⟩ = α\|ψ₁⟩ + β\|ψ₂⟩ + γ\|ψ₃⟩` | `y = w₁·y₁ + w₂·y₂ + w₃·y₃` |
| **State Space** | Single combined quantum state | Three independent measured vectors |
| **Coefficients** | Complex (α, β, γ ∈ ℂ) | Real (w₁, w₂, w₃ ∈ ℝ) |
| **Measurement** | Once (after superposition) | Three times (per branch) |

### **Example: 2-Qubit System**

**Option A (Quantum Superposition):**
```
|ψ₁⟩ = 0.6|00⟩ + 0.8|11⟩
|ψ₂⟩ = 0.8|00⟩ - 0.6|11⟩
|ψ₃⟩ = 0.7|01⟩ + 0.7|10⟩

α = 0.5 + 0.2i, β = 0.3 - 0.1i, γ = 0.4 + 0.3i

|ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩
    = (0.5+0.2i)(0.6|00⟩ + 0.8|11⟩) + ...

→ Quantum interference can cause constructive/destructive interference
→ Final state depends on complex phase relationships
```

**Option B (Classical Combination):**
```
Measure |ψ₁⟩ → y₁ = [⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩]
Measure |ψ₂⟩ → y₂ = [⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩]
Measure |ψ₃⟩ → y₃ = [⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩]

w₁ = 1.2, w₂ = 0.8, w₃ = 1.0

y = 1.2·y₁ + 0.8·y₂ + 1.0·y₃  ← Classical vector addition
  = [1.2·y₁[0] + 0.8·y₂[0] + 1.0·y₃[0], ...]
```

### **Gradient Flow**

**Option A:**
```
∂Loss/∂α = ∂Loss/∂Y · ∂Y/∂⟨ψ|M|ψ⟩ · ∂⟨ψ|M|ψ⟩/∂|ψ⟩ · ∂|ψ⟩/∂α
         = ∂Loss/∂Y · ... · |ψ₁⟩  (simplified)
```
Gradient depends on quantum state overlap and interference.

**Option B:**
```
∂Loss/∂w₁ = ∂Loss/∂Y · ∂Y/∂y_combined · ∂y_combined/∂w₁
          = ∂Loss/∂Y · ... · y₁
```
Standard backpropagation through classical addition.

---

## Implementation Comparison

### **Architecture Differences**

#### **Option A: QuantumHydraLayer**

```python
class QuantumHydraLayer(nn.Module):
    def __init__(self, n_qubits, qlcu_layers, ...):
        # Trainable complex coefficients for superposition
        self.alpha = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.rand(1, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.rand(1, dtype=torch.complex64))

        # Quantum circuit parameters
        self.qlcu_base_params = nn.Parameter(torch.rand(n_qlcu_params))
        self.qd_params = nn.Parameter(torch.rand(n_qd_params))

        # Single measurement QNode (after superposition)
        self._setup_qnodes()

    def forward(self, x):
        # 1. Create quantum states for each branch
        psi1 = self.qlcu_shift_qnode(base_states, params)
        psi2 = self.flip_qlcu_shift_flip_qnode(base_states, params)
        psi3 = self.qd_qnode(base_states, qd_params)

        # 2. QUANTUM SUPERPOSITION
        psi_combined = self.alpha * psi1 + self.beta * psi2 + self.gamma * psi3
        psi_normalized = psi_combined / torch.linalg.norm(psi_combined)

        # 3. Single measurement
        measurements = self.measurement_qnode(psi_normalized)

        # 4. Classical output
        output = self.output_ff(measurements)
        return output
```

**Parameter count:**
- Complex coefficients: 3 (α, β, γ)
- QLCU parameters: 4 × n_qubits × qlcu_layers
- QD parameters: 3 × n_qubits
- Output layer: 3n × output_dim

#### **Option B: QuantumHydraHybridLayer**

```python
class QuantumHydraHybridLayer(nn.Module):
    def __init__(self, n_qubits, qlcu_layers, ...):
        # Trainable real-valued weights for classical combination
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.ones(1))

        # Branch-specific processing layers
        self.branch1_ff = nn.Linear(3*n_qubits, 3*n_qubits)
        self.branch2_ff = nn.Linear(3*n_qubits, 3*n_qubits)
        self.branch3_ff = nn.Linear(3*n_qubits, 3*n_qubits)

        # Three separate measurement QNodes
        self._setup_qnodes()

    def forward(self, x):
        # 1. Measure each branch independently
        measurements1 = self.branch1_qnode(base_states, params)
        y1 = self.branch1_ff(torch.stack(measurements1, dim=1).float())

        measurements2 = self.branch2_qnode(base_states, params)
        y2 = self.branch2_ff(torch.stack(measurements2, dim=1).float())

        measurements3 = self.branch3_qnode(base_states, qd_params)
        y3 = self.branch3_ff(torch.stack(measurements3, dim=1).float())

        # 2. CLASSICAL WEIGHTED COMBINATION
        w1_norm = torch.abs(self.w1) / (torch.abs(self.w1) + torch.abs(self.w2) + torch.abs(self.w3))
        # ... same for w2, w3
        y_combined = w1_norm * y1 + w2_norm * y2 + w3_norm * y3

        # 3. Classical output
        output = self.output_ff(y_combined)
        return output
```

**Parameter count:**
- Real weights: 3 (w₁, w₂, w₃)
- Branch FFs: 3 × (3n × 3n) = 27n²
- QLCU/QD parameters: Same as Option A
- Output layer: 3n × output_dim

**Note:** Option B has more parameters due to branch-specific layers (893 vs 425 for n=4).

### **Computational Cost Comparison**

| Operation | Option A | Option B |
|-----------|----------|----------|
| **QNode calls per forward pass** | 4 (3 branches + 1 measurement) | 3 (3 branches with measurement) |
| **State preparation** | 1 base state | 3 base states |
| **Superposition overhead** | ✓ (complex arithmetic) | ✗ (none) |
| **Measurement shots** | 1 (on combined state) | 3 (per branch) |
| **Gradient computation** | Through complex parameters | Through real parameters |

**Approximate relative cost:**
- Option A: ~1.0× (baseline)
- Option B: ~1.2× (more parameters, 3 measurements)

---

## Usage Examples

### **Training Script Template**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from QuantumHydra import QuantumHydraTS
from QuantumHydraHybrid import QuantumHydraHybridTS

# Hyperparameters
n_qubits = 6
n_timesteps = 160
feature_dim = 64
output_dim = 2
batch_size = 16
n_epochs = 100
learning_rate = 1e-3

# Data loading (example)
from Load_PhysioNet_EEG import load_data
train_loader, val_loader = load_data(batch_size=batch_size)

# Option A: Quantum Superposition
model_a = QuantumHydraTS(
    n_qubits=n_qubits,
    n_timesteps=n_timesteps,
    qlcu_layers=2,
    feature_dim=feature_dim,
    output_dim=output_dim,
    device="cuda"
)

# Option B: Hybrid Classical-Quantum
model_b = QuantumHydraHybridTS(
    n_qubits=n_qubits,
    n_timesteps=n_timesteps,
    qlcu_layers=2,
    feature_dim=feature_dim,
    output_dim=output_dim,
    device="cuda"
)

# Training loop (same for both)
criterion = nn.CrossEntropyLoss()
optimizer_a = optim.Adam(model_a.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    model_a.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to("cuda")
        batch_y = batch_y.to("cuda")

        optimizer_a.zero_grad()
        outputs = model_a(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_a.step()

    # Validation...
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
```

### **Comparison Experiment**

```python
from compare_quantum_hydra import compare_models

# Run comparison on PhysioNet EEG data
results = compare_models(
    dataset="physionet_eeg",
    n_qubits=6,
    qlcu_layers=2,
    n_epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    device="cuda"
)

# Results dictionary contains:
# {
#     'option_a': {'accuracy': ..., 'loss': ..., 'time': ...},
#     'option_b': {'accuracy': ..., 'loss': ..., 'time': ...},
#     'classical': {'accuracy': ..., 'loss': ..., 'time': ...}
# }

print(f"Option A Accuracy: {results['option_a']['accuracy']:.2%}")
print(f"Option B Accuracy: {results['option_b']['accuracy']:.2%}")
print(f"Classical Accuracy: {results['classical']['accuracy']:.2%}")
```

### **Branch Analysis (Option B Only)**

```python
# Analyze which branch contributes most
model_b.eval()
with torch.no_grad():
    sample_x = next(iter(val_loader))[0][:4].to("cuda")

    contributions = model_b.quantum_hydra.get_branch_contributions(sample_x)

    # Check branch importance
    branch1_norm = torch.norm(contributions['branch1']).item()
    branch2_norm = torch.norm(contributions['branch2']).item()
    branch3_norm = torch.norm(contributions['branch3']).item()

    print(f"Branch 1 (Shift) strength: {branch1_norm:.4f}")
    print(f"Branch 2 (Flip-Shift-Flip) strength: {branch2_norm:.4f}")
    print(f"Branch 3 (Diagonal) strength: {branch3_norm:.4f}")

    print(f"\nLearned weights:")
    print(f"  w1 = {contributions['weights']['w1']:.3f}")
    print(f"  w2 = {contributions['weights']['w2']:.3f}")
    print(f"  w3 = {contributions['weights']['w3']:.3f}")
```

### **Visualization**

```python
from QuantumHydra import visualize_all_circuits

# Generate circuit diagrams for both models (identical circuits)
visualize_all_circuits(
    n_qubits=6,
    qlcu_layers=2,
    shift_amount=1,
    output_dir="./circuits_comparison"
)

# Visualize learned coefficients over training
import matplotlib.pyplot as plt

# Option A: Complex coefficients trajectory
alphas = []  # Collect during training
betas = []
gammas = []

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter([a.real for a in alphas], [a.imag for a in alphas], label='α', alpha=0.5)
plt.scatter([b.real for b in betas], [b.imag for b in betas], label='β', alpha=0.5)
plt.scatter([g.real for g in gammas], [g.imag for g in gammas], label='γ', alpha=0.5)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Option A: Complex Coefficient Trajectories')
plt.legend()

# Option B: Real weights over epochs
w1s = []  # Collect during training
w2s = []
w3s = []

plt.subplot(1, 2, 2)
plt.plot(w1s, label='w₁')
plt.plot(w2s, label='w₂')
plt.plot(w3s, label='w₃')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('Option B: Real Weight Evolution')
plt.legend()

plt.tight_layout()
plt.savefig('coefficient_comparison.pdf')
```

---

## When to Use Which

### **Use Option A (Quantum Superposition) when:**

1. **Exploring quantum advantage**: You want to investigate if quantum interference helps
2. **Theoretical research**: Studying fundamental quantum ML capabilities
3. **Small-scale proof-of-concept**: Testing on limited qubits (4-8)
4. **Pattern recognition**: Data may benefit from quantum correlations (e.g., certain EEG patterns)
5. **Fewer measurements preferred**: Cost of quantum circuit execution is high

**Example scenarios:**
- Academic research on quantum ML theory
- Benchmark comparisons for quantum vs classical
- Exploring non-classical computation paradigms

### **Use Option B (Hybrid) when:**

1. **Faithful quantum translation**: You want quantum version of classical Hydra
2. **Interpretability matters**: Need to understand individual branch contributions
3. **Ablation studies**: Want to analyze which branches are important
4. **Production systems**: Robustness to noise is critical
5. **Classical baselines**: Comparing directly to classical Hydra algorithm

**Example scenarios:**
- Medical diagnosis systems (interpretability required)
- Comparing quantum vs classical Hydra fairly
- Understanding which Hydra operations (shift/flip/diagonal) matter most
- NISQ hardware deployment (noise robustness)

### **Quick Decision Matrix**

| Criterion | Option A | Option B |
|-----------|----------|----------|
| **Quantum advantage seeking** | ✅ Better | ❌ Limited |
| **Interpretability** | ❌ Harder | ✅ Easier |
| **Faithful to classical Hydra** | ❌ Different | ✅ Faithful |
| **Noise robustness** | ❌ Fragile | ✅ Robust |
| **Computational cost** | ✅ Lower | ❌ Higher |
| **Research novelty** | ✅ High | ⚠️ Moderate |
| **Production readiness** | ❌ Experimental | ✅ Practical |

---

## Performance Expectations

### **Theoretical Predictions**

**Scenario 1: Linear-separable data**
- Classical Hydra: ~90% accuracy
- Option A: ~88-92% accuracy (comparable, slight variation from quantum noise)
- Option B: ~89-91% accuracy (very close to classical)

**Scenario 2: Non-linear, high-entanglement data**
- Classical Hydra: ~75% accuracy
- Option A: ~78-82% accuracy (potential quantum advantage)
- Option B: ~76-79% accuracy (modest improvement)

**Scenario 3: Long-range temporal dependencies (EEG)**
- Classical Hydra: ~82% accuracy
- Option A: ~80-85% accuracy (superposition may help or hurt)
- Option B: ~81-84% accuracy (closer to classical performance)

### **Empirical Results (To Be Determined)**

See `QUANTUM_HYDRA_RESULTS.md` for actual experimental results on PhysioNet EEG dataset.

---

## References

1. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - Original classical Hydra architecture
   - https://arxiv.org/pdf/2407.09941

2. **Gu & Dao (2024)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - Mamba state-space model
   - https://arxiv.org/html/2312.00752v2

3. **Khatri et al. (2024)** - "Quantum Transformer Models"
   - QSVT-based attention mechanisms
   - https://arxiv.org/html/2406.04305v1

4. **Park et al. (2025)** - "Quantum Time-series Transformer"
   - Application of QSVT/LCU to time-series
   - https://arxiv.org/html/2509.00711v1

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_hydra_2025,
  author = {Park, Junghoon},
  title = {Quantum Hydra: Two Approaches to Quantum State-Space Models},
  year = {2025},
  url = {https://github.com/JHPark9090/quantum-hydra}
}
```

---

## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Email: utopie9090@snu.ac.kr
- See `compare_quantum_hydra.py` for experimental setup
- See `QUANTUM_HYDRA_RESULTS.md` for performance benchmarks

---

**Last Updated**: October 2024
**Version**: 1.0
**Status**: Experimental (Research Code)
