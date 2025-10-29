import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Import models
from QuantumHydra import QuantumHydraTS
from QuantumHydraHybrid import QuantumHydraHybridTS
from TrueClassicalHydra import TrueClassicalHydra
from TrueClassicalMamba import TrueClassicalMamba

# Import data loader
from Load_PhysioNet_EEG_NoPrompt import load_eeg_ts_revised

"""
Comprehensive comparison of four state-space models:

QUANTUM MODELS:
1. Quantum Hydra (Superposition) - Option A from QUANTUM_HYDRA_README.md
2. Quantum Hydra Hybrid - Option B from QUANTUM_HYDRA_README.md

CLASSICAL BASELINES (CORRECTED):
3. True Classical Hydra - Faithful implementation of Hwang et al. (2024)
4. True Classical Mamba - Faithful implementation of Gu & Dao (2024)

Experiments conducted on PhysioNet EEG Motor Imagery dataset.
"""


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate additional metrics
    try:
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, auc, f1, cm


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    model_name,
    n_epochs=50,
    learning_rate=1e-3,
    device="cuda",
    verbose=True
):
    """
    Train a single model and return results.

    Returns:
        dict with training history and final test metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'epochs': []
    }

    best_val_acc = 0
    best_model_state = None
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_auc, val_f1, _ = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['epochs'].append(epoch + 1)

        epoch_time = time.time() - epoch_start

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} ({epoch_time:.2f}s) | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    total_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc, test_f1, test_cm = evaluate(model, test_loader, criterion, device)

    print(f"\n{model_name} Training Complete!")
    print(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    print(f"  Confusion Matrix:\n{test_cm}")

    results = {
        'model_name': model_name,
        'history': history,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'training_time': total_time,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate
    }

    return results


def plot_comparison(results_dict, save_path="all_models_comparison.pdf"):
    """
    Plot training curves and final metrics comparison for all four models.

    Args:
        results_dict: Dictionary with keys 'quantum_super', 'quantum_hybrid', 'hydra', 'mamba'
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    colors = {
        'quantum_super': 'blue',
        'quantum_hybrid': 'green',
        'hydra': 'red',
        'mamba': 'purple'
    }

    labels = {
        'quantum_super': 'Quantum Hydra (Superposition)',
        'quantum_hybrid': 'Quantum Hydra (Hybrid)',
        'hydra': 'Classical Hydra',
        'mamba': 'Classical Mamba'
    }

    markers = {
        'quantum_super': 'o',
        'quantum_hybrid': 's',
        'hydra': '^',
        'mamba': 'D'
    }

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for key, results in results_dict.items():
        history = results['history']
        ax.plot(history['epochs'], history['train_loss'],
                color=colors[key], label=labels[key], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for key, results in results_dict.items():
        history = results['history']
        ax.plot(history['epochs'], history['val_loss'],
                color=colors[key], label=labels[key], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax = axes[0, 2]
    for key, results in results_dict.items():
        history = results['history']
        ax.plot(history['epochs'], history['val_acc'],
                color=colors[key], label=labels[key], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Test Accuracy Comparison (Bar Chart)
    ax = axes[1, 0]
    test_accs = [results_dict[key]['test_acc'] for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']]
    bars = ax.bar(['Q-Super', 'Q-Hybrid', 'Hydra', 'Mamba'], test_accs,
                   color=[colors[k] for k in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']])
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy')
    ax.set_ylim([0, 1])
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Test AUC Comparison (Bar Chart)
    ax = axes[1, 1]
    test_aucs = [results_dict[key]['test_auc'] for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']]
    bars = ax.bar(['Q-Super', 'Q-Hybrid', 'Hydra', 'Mamba'], test_aucs,
                   color=[colors[k] for k in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']])
    ax.set_ylabel('Test AUC')
    ax.set_title('Final Test AUC')
    ax.set_ylim([0, 1])
    for bar, auc in zip(bars, test_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Test F1 Score Comparison (Bar Chart)
    ax = axes[1, 2]
    test_f1s = [results_dict[key]['test_f1'] for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']]
    bars = ax.bar(['Q-Super', 'Q-Hybrid', 'Hydra', 'Mamba'], test_f1s,
                   color=[colors[k] for k in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']])
    ax.set_ylabel('Test F1 Score')
    ax.set_title('Final Test F1 Score')
    ax.set_ylim([0, 1])
    for bar, f1 in zip(bars, test_f1s):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 7: Training Time Comparison (Bar Chart)
    ax = axes[2, 0]
    train_times = [results_dict[key]['training_time']/60 for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']]
    bars = ax.bar(['Q-Super', 'Q-Hybrid', 'Hydra', 'Mamba'], train_times,
                   color=[colors[k] for k in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']])
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison')
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 8: Model Parameters Comparison (Bar Chart)
    ax = axes[2, 1]
    # This will be filled during runtime
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Size Comparison')
    ax.text(0.5, 0.5, 'Parameters logged in console',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 9: Performance vs Time Trade-off
    ax = axes[2, 2]
    for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']:
        ax.scatter(results_dict[key]['training_time']/60,
                  results_dict[key]['test_acc'],
                  color=colors[key],
                  s=200,
                  marker=markers[key],
                  label=labels[key],
                  alpha=0.7,
                  edgecolors='black',
                  linewidths=1.5)
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def run_comprehensive_comparison(
    n_qubits=4,
    qlcu_layers=2,
    hidden_dim=64,
    d_model=128,
    d_state=16,
    n_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    sample_size=10,
    sampling_freq=100,
    seed=2024,
    device="cuda"
):
    """
    Run comprehensive comparison experiment for all four models.

    Args:
        n_qubits: Number of qubits for quantum models
        qlcu_layers: QLCU circuit depth
        hidden_dim: Hidden dimension for Hydra
        d_model: Model dimension for Mamba
        d_state: State dimension for Mamba
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sample_size: Number of PhysioNet subjects
        sampling_freq: EEG sampling frequency
        seed: Random seed
        device: 'cuda' or 'cpu'

    Returns:
        dict: Results for all four models
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*80)
    print("Loading PhysioNet EEG Data")
    print("="*80)

    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=seed,
        device=device,
        batch_size=batch_size,
        sampling_freq=sampling_freq,
        sample_size=sample_size
    )

    n_channels = input_dim[1]
    n_timesteps = input_dim[2]
    output_dim = 2  # Binary classification (left/right)

    print(f"\nData loaded successfully!")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Channels: {n_channels}, Timesteps: {n_timesteps}")
    print(f"  Output classes: {output_dim}")

    results = {}

    # ========================================================================
    # Model 1: Quantum Hydra (Quantum Superposition) - Option A
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Quantum Hydra (Superposition) - Option A")
    print("="*80)
    print("Based on quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩ + γ|ψ₃⟩")

    model_quantum_super = QuantumHydraTS(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        shift_amount=1,
        feature_dim=n_channels,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    ).to(device)

    params_quantum_super = sum(p.numel() for p in model_quantum_super.parameters())
    print(f"Model parameters: {params_quantum_super:,}")

    results['quantum_super'] = train_model(
        model=model_quantum_super,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Quantum Hydra (Superposition)",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )
    results['quantum_super']['n_params'] = params_quantum_super

    # ========================================================================
    # Model 2: Quantum Hydra Hybrid (Classical Combination) - Option B
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Quantum Hydra Hybrid - Option B")
    print("="*80)
    print("Based on classical combination: y = w₁·y₁ + w₂·y₂ + w₃·y₃")

    model_quantum_hybrid = QuantumHydraHybridTS(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        shift_amount=1,
        feature_dim=n_channels,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    ).to(device)

    params_quantum_hybrid = sum(p.numel() for p in model_quantum_hybrid.parameters())
    print(f"Model parameters: {params_quantum_hybrid:,}")

    results['quantum_hybrid'] = train_model(
        model=model_quantum_hybrid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Quantum Hydra (Hybrid)",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )
    results['quantum_hybrid']['n_params'] = params_quantum_hybrid

    # ========================================================================
    # Model 3: True Classical Hydra (Corrected Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: True Classical Hydra - Corrected Baseline")
    print("="*80)
    print("Faithful implementation of Hwang et al. (2024)")

    model_hydra = TrueClassicalHydra(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        n_hydra_layers=2,
        output_dim=output_dim,
        dropout=0.1
    ).to(device)

    params_hydra = sum(p.numel() for p in model_hydra.parameters())
    print(f"Model parameters: {params_hydra:,}")

    results['hydra'] = train_model(
        model=model_hydra,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="True Classical Hydra",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )
    results['hydra']['n_params'] = params_hydra

    # ========================================================================
    # Model 4: True Classical Mamba (Additional Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 4: True Classical Mamba - Additional Baseline")
    print("="*80)
    print("Faithful implementation of Gu & Dao (2024)")

    model_mamba = TrueClassicalMamba(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        d_model=d_model,
        d_state=d_state,
        n_layers=2,
        output_dim=output_dim,
        dropout=0.1
    ).to(device)

    params_mamba = sum(p.numel() for p in model_mamba.parameters())
    print(f"Model parameters: {params_mamba:,}")

    results['mamba'] = train_model(
        model=model_mamba,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="True Classical Mamba",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )
    results['mamba']['n_params'] = params_mamba

    # ========================================================================
    # Summary and Visualization
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Model':<40} {'Params':<12} {'Test Acc':<12} {'Test AUC':<12} {'Test F1':<12} {'Time (min)':<12}")
    print("-" * 108)

    for key in ['quantum_super', 'quantum_hybrid', 'hydra', 'mamba']:
        r = results[key]
        print(f"{r['model_name']:<40} {r['n_params']:<12,} {r['test_acc']:<12.4f} {r['test_auc']:<12.4f} "
              f"{r['test_f1']:<12.4f} {r['training_time']/60:<12.2f}")

    print("\n" + "="*80)
    print("PERFORMANCE RANKING (by Test Accuracy)")
    print("="*80)

    ranked = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    for rank, (key, r) in enumerate(ranked, 1):
        print(f"{rank}. {r['model_name']}: {r['test_acc']:.4f} ({r['test_acc']*100:.2f}%)")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"all_models_comparison_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = value.copy()

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(results, save_path=f"all_models_comparison_{timestamp}.pdf")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of Quantum Hydra models vs Classical baselines"
    )

    # Model hyperparameters
    parser.add_argument("--n-qubits", type=int, default=4,
                       help="Number of qubits for quantum models")
    parser.add_argument("--qlcu-layers", type=int, default=2,
                       help="QLCU circuit depth")
    parser.add_argument("--hidden-dim", type=int, default=64,
                       help="Hidden dimension for classical Hydra")
    parser.add_argument("--d-model", type=int, default=128,
                       help="Model dimension for Mamba")
    parser.add_argument("--d-state", type=int, default=16,
                       help="State dimension for Mamba")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")

    # Data parameters
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Number of PhysioNet subjects")
    parser.add_argument("--sampling-freq", type=int, default=100,
                       help="EEG sampling frequency (Hz)")
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed for reproducibility")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    print("\nModels to be compared:")
    print("  1. Quantum Hydra (Superposition) - Option A")
    print("  2. Quantum Hydra (Hybrid) - Option B")
    print("  3. True Classical Hydra - Corrected baseline")
    print("  4. True Classical Mamba - Additional baseline")
    print("\nDataset: PhysioNet EEG Motor Imagery")
    print("="*80)

    results = run_comprehensive_comparison(
        n_qubits=args.n_qubits,
        qlcu_layers=args.qlcu_layers,
        hidden_dim=args.hidden_dim,
        d_model=args.d_model,
        d_state=args.d_state,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        sampling_freq=args.sampling_freq,
        seed=args.seed,
        device=args.device
    )

    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"  - Best performing model: {max(results.items(), key=lambda x: x[1]['test_acc'])[1]['model_name']}")
    print(f"  - Fastest training: {min(results.items(), key=lambda x: x[1]['training_time'])[1]['model_name']}")
    print(f"  - Most parameters: {max(results.items(), key=lambda x: x[1]['n_params'])[1]['model_name']}")
    print("="*80)
