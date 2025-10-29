# Quantum Mamba Models - Comprehensive Documentation

> Quantum implementations of the Mamba architecture for ablation studies

## 🎯 Overview

This document describes **two quantum implementations of the Mamba architecture** (Gu & Dao, 2024) designed for ablation studies comparing quantum and classical state-space models.

### Why Quantum Mamba?

1. **Ablation Study**: Compare Quantum Hydra vs Quantum Mamba to determine if quantum advantages are architecture-specific
2. **Mamba Architecture**: State-of-the-art selective SSM with input-dependent parameters
3. **Two Design Options**: Test both superposition (Option A) and hybrid (Option B) approaches

---

## 📊 The Two Quantum Mamba Models

### **Option A: Quantum Mamba (Superposition)** 🔵

**File:** `QuantumMamba.py`

**Mathematical Formulation:**
```
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
where α, β, γ ∈ ℂ (complex coefficients)

|ψ_ssm⟩  = Q_SelectiveSSM|X⟩    (state-space path)
|ψ_gate⟩ = Q_Gating|X⟩          (gating path)
|ψ_skip⟩ = Q_Skip|X⟩            (skip connection)
```

**Key Features:**
- **Quantum superposition** of three pathways before measurement
- **Complex-valued trainable coefficients** (α, β, γ)
- **Single measurement** on combined quantum state
- **Potential for quantum interference** between SSM, gating, and skip paths
- **Fewer quantum circuit calls** per forward pass

**Advantages:**
- ✓ Quantum interference may capture non-classical correlations
- ✓ True quantum superposition (not just classical mixing)
- ✓ Exponential state space (2^n)
- ✓ Fewer measurements required

**Limitations:**
- ✗ Different semantics from classical Mamba
- ✗ Sensitive to quantum decoherence
- ✗ Complex coefficient optimization
- ✗ Harder to interpret

**Code Structure:**
```python
class QuantumMambaLayer(nn.Module):
    def __init__(self, n_qubits, qlcu_layers, ...):
        # Complex coefficients for superposition
        self.alpha_real = nn.Parameter(torch.rand(1))
        self.alpha_imag = nn.Parameter(torch.zeros(1))
        self.beta_real = nn.Parameter(torch.rand(1))
        self.beta_imag = nn.Parameter(torch.zeros(1))
        self.gamma_real = nn.Parameter(torch.rand(1))
        self.gamma_imag = nn.Parameter(torch.zeros(1))

        # Three quantum paths
        self.ssm_path = QuantumSelectiveSSM(...)
        self.gate_path = QuantumGatingPath(...)
        self.skip_path = QuantumSkipPath(...)

    def forward(self, x):
        # Compute three quantum states
        psi1 = self.ssm_path(x)   # SSM
        psi2 = self.gate_path(x)  # Gating
        psi3 = self.skip_path(x)  # Skip

        # QUANTUM SUPERPOSITION
        alpha = torch.complex(self.alpha_real, self.alpha_imag)
        beta = torch.complex(self.beta_real, self.beta_imag)
        gamma = torch.complex(self.gamma_real, self.gamma_imag)

        psi_combined = alpha * psi1 + beta * psi2 + gamma * psi3
        psi_normalized = psi_combined / norm(psi_combined)

        # Measure
        return output_layer(abs(psi_normalized))
```

---

### **Option B: Quantum Mamba (Hybrid)** 🟢

**File:** `QuantumMambaHybrid.py`

**Mathematical Formulation:**
```
y₁ = Measure(Q_SelectiveSSM|X⟩)
y₂ = Measure(Q_Gating|X⟩)
y₃ = Measure(Q_Skip|X⟩)

Y = OutputLayer(w₁·y₁ + w₂·y₂ + w₃·y₃)
where w₁, w₂, w₃ ∈ ℝ (real weights)
```

**Key Features:**
- **Three independent quantum circuits**
- **Classical weighted combination** of measurements
- **Real-valued trainable weights**
- **Faithful to classical Mamba's addition semantics**
- **More interpretable** branch contributions

**Advantages:**
- ✓ Preserves classical Mamba semantics
- ✓ Interpretable (can analyze each branch)
- ✓ More robust to quantum noise
- ✓ Real-valued weights (easier optimization)
- ✓ Branch contribution analysis available

**Limitations:**
- ✗ No quantum interference
- ✗ More quantum circuit calls (three separate circuits)
- ✗ Higher computational cost

**Code Structure:**
```python
class QuantumMambaHybridLayer(nn.Module):
    def __init__(self, n_qubits, qlcu_layers, ...):
        # Real-valued weights
        self.w1 = nn.Parameter(torch.ones(1))  # SSM weight
        self.w2 = nn.Parameter(torch.ones(1))  # Gate weight
        self.w3 = nn.Parameter(torch.ones(1))  # Skip weight

        # Branch-specific processing
        self.branch1_ff = nn.Linear(...)  # SSM path
        self.branch2_ff = nn.Linear(...)  # Gate path
        self.branch3_ff = nn.Linear(...)  # Skip path

    def forward(self, x):
        # Measure each branch independently
        y1 = measure(self.branch1_qnode(x))  # SSM
        y2 = measure(self.branch2_qnode(x))  # Gate
        y3 = measure(self.branch3_qnode(x))  # Skip

        # CLASSICAL WEIGHTED COMBINATION
        w_sum = abs(w1) + abs(w2) + abs(w3)
        y_combined = (w1/w_sum)*y1 + (w2/w_sum)*y2 + (w3/w_sum)*y3

        return output_layer(y_combined)
```

---

## 🔬 Quantum Circuit Components

### 1. Quantum Selective SSM Circuit

Implements the selective state-space model with **input-dependent B, C, dt parameters**.

```python
def selective_ssm_circuit(qlcu_params, b_params, c_params, dt_params):
    """
    Quantum implementation of selective SSM.

    Classical Mamba SSM:
        h[t] = A * h[t-1] + B(u) * u[t]
        y[t] = C(u) * h[t] + D * u[t]

    Quantum implementation:
        - QLCU: Simulates A matrix (state evolution)
        - b_params: Input-dependent B matrix
        - c_params: Input-dependent C matrix
        - dt_params: Time-step parameters
    """
    # A matrix transformation (via QLCU)
    for layer in range(qlcu_layers):
        qml.RY(qlcu_params)
        qml.IsingXX(qlcu_params)

        # Input-dependent B matrix
        qml.RY(b_params)

        qml.RY(qlcu_params)
        qml.IsingYY(qlcu_params)

        # Output-dependent C matrix
        qml.RZ(c_params)

        # Time-step modulation
        qml.RX(dt_params)
```

**Key features:**
- ✓ Input-dependent parameters (B, C, dt)
- ✓ QLCU for state transformation
- ✓ Time-step modulation
- ✓ Entanglement for state mixing

---

### 2. Quantum Gating Circuit

Implements the multiplicative gating mechanism (like SiLU gating in Mamba).

```python
def gating_circuit(gate_params):
    """
    Quantum gating mechanism.

    Classical Mamba:
        output = y * silu(z)

    Quantum implementation:
        Controlled gates for multiplicative gating
    """
    for layer in range(gate_layers):
        qml.RX(gate_params)
        qml.CRZ(gate_params)  # Gating via controlled rotations
        qml.RY(gate_params)
        qml.CRZ(gate_params)
```

**Key features:**
- ✓ Controlled gates for gating
- ✓ Ring connectivity
- ✓ Mimics SiLU gating behavior

---

### 3. Quantum Skip Circuit

Implements skip connection (D matrix in Mamba).

```python
def skip_circuit(skip_params):
    """
    Quantum skip connection.

    Classical Mamba:
        y[t] += D * u[t]

    Quantum implementation:
        Diagonal operations for direct path
    """
    qml.RX(skip_params)
    qml.RY(skip_params)
    qml.RZ(skip_params)
```

**Key features:**
- ✓ Diagonal operations
- ✓ Independent on each qubit
- ✓ Direct input-to-output path

---

## 📈 Model Architectures

### QuantumMambaLayer (Option A)

```
Input (batch, feature_dim)
    ↓
Feature Projection → (batch, n_params)
    ↓
Three Quantum Paths:
    ├─ QuantumSelectiveSSM → |ψ_ssm⟩
    ├─ QuantumGatingPath   → |ψ_gate⟩
    └─ QuantumSkipPath     → |ψ_skip⟩
    ↓
QUANTUM SUPERPOSITION
    |ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩
    ↓
Normalization
    ↓
Measurement (|ψ|)
    ↓
Output Layer → (batch, output_dim)
```

### QuantumMambaHybridLayer (Option B)

```
Input (batch, feature_dim)
    ↓
Feature Projection → (batch, n_params)
    ↓
Three Independent Quantum Circuits:
    ├─ Branch 1 QNode → Measure → y₁
    ├─ Branch 2 QNode → Measure → y₂
    └─ Branch 3 QNode → Measure → y₃
    ↓
CLASSICAL WEIGHTED COMBINATION
    y = w₁·y₁ + w₂·y₂ + w₃·y₃
    ↓
Output Layer → (batch, output_dim)
```

### QuantumMambaTS (Time-Series)

```
Input (batch, channels, timesteps)
    ↓
Temporal Conv1d (like Mamba)
    ↓
Adaptive Pooling
    ↓
Feature Projection
    ↓
QuantumMamba Layer (Option A or B)
    ↓
Output Layer → (batch, output_dim)
```

---

## 🚀 Usage

### Quick Start

```python
import torch
from QuantumMamba import QuantumMambaLayer, QuantumMambaTS
from QuantumMambaHybrid import QuantumMambaHybridLayer, QuantumMambaHybridTS

# Option A: Superposition
model_a = QuantumMambaTS(
    n_qubits=6,
    n_timesteps=200,
    qlcu_layers=2,
    gate_layers=2,
    feature_dim=129,
    output_dim=2,
    device="cuda"
)

# Option B: Hybrid
model_b = QuantumMambaHybridTS(
    n_qubits=6,
    n_timesteps=200,
    qlcu_layers=2,
    gate_layers=2,
    feature_dim=129,
    output_dim=2,
    device="cuda"
)

# Forward pass
x = torch.randn(16, 129, 200)  # (batch, channels, timesteps)
output_a = model_a(x)  # (batch, 2)
output_b = model_b(x)  # (batch, 2)
```

### Branch Contribution Analysis (Option B only)

```python
# Analyze which branch contributes most
contributions = model_b.quantum_mamba.get_branch_contributions(x[:4])

print(f"SSM branch shape: {contributions['branch1_ssm'].shape}")
print(f"Gate branch shape: {contributions['branch2_gate'].shape}")
print(f"Skip branch shape: {contributions['branch3_skip'].shape}")
print(f"Weights: SSM={contributions['weights']['w1_ssm']:.3f}, "
      f"Gate={contributions['weights']['w2_gate']:.3f}, "
      f"Skip={contributions['weights']['w3_skip']:.3f}")
```

### Training Example

```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = QuantumMambaTS(n_qubits=6, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 🔍 Comparison with Quantum Hydra

| Aspect | Quantum Hydra | Quantum Mamba |
|--------|---------------|---------------|
| **Base Architecture** | Bidirectional Hydra (shift/flip) | Selective SSM (input-dependent) |
| **Key Mechanism** | Shift, Flip, Diagonal | Selective B/C/dt, Gating, Skip |
| **Classical Paper** | Hwang et al. 2024 | Gu & Dao 2024 |
| **Branch 1** | Qshift(QLCU\|X⟩) | Q_SelectiveSSM\|X⟩ |
| **Branch 2** | Qflip(Qshift(QLCU(Qflip\|X⟩))) | Q_Gating\|X⟩ |
| **Branch 3** | QD\|X⟩ | Q_Skip\|X⟩ |
| **Superposition** | α\|ψ₁⟩ + β\|ψ₂⟩ + γ\|ψ₃⟩ | α\|ψ_ssm⟩ + β\|ψ_gate⟩ + γ\|ψ_skip⟩ |
| **Hybrid** | w₁·y₁ + w₂·y₂ + w₃·y₃ | w₁·y₁ + w₂·y₂ + w₃·y₃ |

---

## 🧪 Ablation Study Design

### Research Questions

1. **Architecture-Specific vs General Quantum Advantage**
   - Does quantum superposition help more in Hydra or Mamba?
   - Are quantum advantages specific to certain SSM architectures?

2. **Superposition vs Hybrid**
   - Option A vs Option B for both Hydra and Mamba
   - Which design benefits more from quantum interference?

3. **Comparison Matrix**

| Model | Type | Quantum | Classical |
|-------|------|---------|-----------|
| Quantum Hydra (Super) | Option A | ✓ | - |
| Quantum Hydra (Hybrid) | Option B | ✓ | - |
| Quantum Mamba (Super) | Option A | ✓ | - |
| Quantum Mamba (Hybrid) | Option B | ✓ | - |
| True Classical Hydra | Baseline | - | ✓ |
| True Classical Mamba | Baseline | - | ✓ |

**Total: 6 models**

### Metrics to Track

- **Accuracy**: Classification performance
- **AUC**: Area under ROC curve
- **F1 Score**: Balanced metric
- **Training Time**: Computational cost
- **Parameters**: Model size
- **Branch Contributions**: Which path is most important (Option B only)

---

## 📊 Expected Results

### Hypotheses

**H1: Quantum advantages are general**
- Both Quantum Hydra and Quantum Mamba outperform classical baselines
- Quantum superposition helps regardless of architecture

**H2: Quantum advantages are architecture-specific**
- Only one of Quantum Hydra/Mamba shows improvements
- Some architectures benefit more from quantum circuits

**H3: Superposition matters**
- Option A (superposition) outperforms Option B (hybrid)
- Quantum interference is crucial for advantages

**H4: Hybrid is more practical**
- Option B (hybrid) more robust to noise
- Easier to train and interpret

---

## 🛠️ Implementation Details

### Parameter Counts

**Option A (Superposition):**
- QLCU params: 4 × n_qubits × qlcu_layers
- Gate params: (3 × n_qubits + n_qubits) × gate_layers
- Skip params: 3 × n_qubits
- Selective params: 3 × n_qubits (B, C, dt)
- Complex coefficients: 6 (real + imag for α, β, γ)

**Option B (Hybrid):**
- Branch 1 params: 4 × n_qubits × qlcu_layers + 3 × n_qubits
- Branch 2 params: (3 × n_qubits + n_qubits) × gate_layers
- Branch 3 params: 3 × n_qubits
- Classical weights: 3 (w₁, w₂, w₃)

### Quantum Backend

- **Framework**: PennyLane
- **Device**: `default.qubit` (noiseless simulation)
- **Differentiation**: `backprop` (auto-diff through quantum circuits)
- **Interface**: `torch` (PyTorch integration)

### Typical Configuration

```python
n_qubits = 6          # 2^6 = 64 dimensional quantum state
qlcu_layers = 2       # Circuit depth for SSM
gate_layers = 2       # Circuit depth for gating
feature_dim = 129     # EEG channels
n_timesteps = 200     # EEG timesteps
output_dim = 2        # Binary classification
batch_size = 16
learning_rate = 1e-3
```

---

## 📝 Key Differences from Classical Mamba

### Classical Mamba (Gu & Dao, 2024)

```python
# Selective SSM
B = Linear(input)     # Input-dependent
C = Linear(input)     # Input-dependent
dt = Softplus(Linear(input))

# State evolution
h[t] = A * h[t-1] + B * u[t]
y[t] = C * h[t] + D * u[t]

# Gating
output = y * silu(gate)
```

### Quantum Mamba (This Implementation)

```python
# Quantum selective SSM
b_params = f(input)   # Input-dependent quantum params
c_params = f(input)
dt_params = f(input)

# Quantum state evolution (via QLCU)
|ψ[t]⟩ = QLCU(b, c, dt) |ψ[t-1]⟩

# Three quantum paths
|ψ_ssm⟩ = Q_SelectiveSSM(b, c, dt) |X⟩
|ψ_gate⟩ = Q_Gating |X⟩
|ψ_skip⟩ = Q_Skip |X⟩

# Superposition (Option A) or Classical (Option B)
|ψ⟩ = α|ψ_ssm⟩ + β|ψ_gate⟩ + γ|ψ_skip⟩  (A)
y = w₁·y₁ + w₂·y₂ + w₃·y₃                (B)
```

---

## ✅ Testing

Both models include comprehensive test suites:

```bash
# Test Option A (Superposition)
python QuantumMamba.py

# Test Option B (Hybrid)
python QuantumMambaHybrid.py
```

**Expected output:**
- ✓ Model initialization
- ✓ Forward pass (correct shapes)
- ✓ Gradient flow verification
- ✓ Parameter count
- ✓ Branch contribution analysis (Option B)

---

## 🔗 Related Files

- `QuantumMamba.py` - Option A implementation
- `QuantumMambaHybrid.py` - Option B implementation
- `TrueClassicalMamba.py` - Classical baseline for comparison
- `QuantumHydra.py` - Quantum Hydra (Option A)
- `QuantumHydraHybrid.py` - Quantum Hydra (Option B)
- `TrueClassicalHydra.py` - Classical Hydra baseline
- `QUANTUM_HYDRA_README.md` - Quantum Hydra documentation
- `COMPARISON_README.md` - Model comparison documentation

---

## 📚 References

### Papers

1. **Gu & Dao (2024)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/html/2312.00752v2
   - Original Mamba architecture

2. **Hwang et al. (2024)** - "Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers"
   - https://arxiv.org/pdf/2407.09941
   - Hydra architecture for comparison

### Code

- Mamba GitHub: https://github.com/state-spaces/mamba
- PennyLane: https://pennylane.ai/

---

## 💡 Design Philosophy

### Why Three Branches?

Both Quantum Hydra and Quantum Mamba use three branches following their classical counterparts:

**Classical Hydra:**
```
QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + DX
        └─ Branch 1─┘   └────── Branch 2 ──────┘   └─3─┘
```

**Classical Mamba:**
```
output = SSM(u) + Gate(u) + Skip(u)
         └─ 1 ─┘   └─ 2 ─┘   └─ 3 ─┘
```

**Quantum versions maintain this structure** to enable fair comparison with classical baselines.

### Why Two Options (A & B)?

- **Option A (Superposition)**: Tests if quantum interference helps
- **Option B (Hybrid)**: More faithful to classical semantics, easier to interpret

By implementing both, we can determine:
1. If quantum advantages exist
2. If they come from interference (A) or quantum circuits alone (B)

---

## 🎯 Next Steps

1. **Run ablation study** with all 6 models
2. **Compare performance** on EEG/fMRI datasets
3. **Analyze branch contributions** (Option B)
4. **Test noise robustness**
5. **Scale to larger qubit counts**

---

**Author:** Junghoon Park
**Created:** October 2024
**Status:** Production-ready
**Purpose:** Ablation study for quantum vs classical SSMs
