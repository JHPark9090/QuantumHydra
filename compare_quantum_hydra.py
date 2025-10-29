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
from ClassicalHydra import ClassicalHydra, ClassicalHydraSimple

# Import data loader
from Load_PhysioNet_EEG_NoPrompt import load_eeg_ts_revised

"""
Comprehensive comparison of three Hydra implementations:
1. Option A: Quantum Hydra (Quantum Superposition)
2. Option B: Quantum Hydra Hybrid (Classical Combination)
3. Classical Hydra baseline

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


def plot_comparison(results_dict, save_path="comparison_results.pdf"):
    """
    Plot training curves and final metrics comparison.

    Args:
        results_dict: Dictionary with keys 'option_a', 'option_b', 'classical'
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'option_a': 'blue',
        'option_b': 'green',
        'classical': 'red'
    }

    labels = {
        'option_a': 'Option A (Quantum Superposition)',
        'option_b': 'Option B (Hybrid)',
        'classical': 'Classical Hydra'
    }

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for key, results in results_dict.items():
        history = results['history']
        ax.plot(history['epochs'], history['train_loss'],
                color=colors[key], label=labels[key], linewidth=2)
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
                color=colors[key], label=labels[key], linewidth=2)
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
                color=colors[key], label=labels[key], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Test Accuracy Comparison (Bar Chart)
    ax = axes[1, 0]
    test_accs = [results_dict[key]['test_acc'] for key in ['option_a', 'option_b', 'classical']]
    bars = ax.bar(['Option A', 'Option B', 'Classical'], test_accs,
                   color=[colors['option_a'], colors['option_b'], colors['classical']])
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
    test_aucs = [results_dict[key]['test_auc'] for key in ['option_a', 'option_b', 'classical']]
    bars = ax.bar(['Option A', 'Option B', 'Classical'], test_aucs,
                   color=[colors['option_a'], colors['option_b'], colors['classical']])
    ax.set_ylabel('Test AUC')
    ax.set_title('Final Test AUC')
    ax.set_ylim([0, 1])
    for bar, auc in zip(bars, test_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Training Time Comparison (Bar Chart)
    ax = axes[1, 2]
    train_times = [results_dict[key]['training_time']/60 for key in ['option_a', 'option_b', 'classical']]
    bars = ax.bar(['Option A', 'Option B', 'Classical'], train_times,
                   color=[colors['option_a'], colors['option_b'], colors['classical']])
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Training Time Comparison')
    for bar, t in zip(bars, train_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.2f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def run_comparison(
    n_qubits=4,
    qlcu_layers=2,
    hidden_dim=64,
    n_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    sample_size=10,
    sampling_freq=100,
    seed=2024,
    device="cuda"
):
    """
    Run comparison experiment for all three models.

    Args:
        n_qubits: Number of qubits for quantum models
        qlcu_layers: QLCU circuit depth
        hidden_dim: Hidden dimension for classical model
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sample_size: Number of PhysioNet subjects
        sampling_freq: EEG sampling frequency
        seed: Random seed
        device: 'cuda' or 'cpu'

    Returns:
        dict: Results for all three models
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
    # Model 1: Option A - Quantum Hydra (Quantum Superposition)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: Option A - Quantum Hydra (Superposition)")
    print("="*80)

    model_a = QuantumHydraTS(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        shift_amount=1,
        feature_dim=n_channels,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model_a.parameters()):,}")

    results['option_a'] = train_model(
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Option A (Quantum Superposition)",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )

    # ========================================================================
    # Model 2: Option B - Quantum Hydra Hybrid (Classical Combination)
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: Option B - Quantum Hydra Hybrid")
    print("="*80)

    model_b = QuantumHydraHybridTS(
        n_qubits=n_qubits,
        n_timesteps=n_timesteps,
        qlcu_layers=qlcu_layers,
        shift_amount=1,
        feature_dim=n_channels,
        output_dim=output_dim,
        dropout=0.1,
        device=device
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    results['option_b'] = train_model(
        model=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Option B (Hybrid Classical-Quantum)",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )

    # ========================================================================
    # Model 3: Classical Hydra Baseline
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 3: Classical Hydra Baseline")
    print("="*80)

    model_c = ClassicalHydra(
        n_channels=n_channels,
        n_timesteps=n_timesteps,
        hidden_dim=hidden_dim,
        n_hydra_layers=2,
        output_dim=output_dim,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model_c.parameters()):,}")

    results['classical'] = train_model(
        model=model_c,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="Classical Hydra",
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )

    # ========================================================================
    # Summary and Visualization
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n{'Model':<40} {'Test Acc':<12} {'Test AUC':<12} {'Test F1':<12} {'Time (min)':<12}")
    print("-" * 88)
    for key in ['option_a', 'option_b', 'classical']:
        r = results[key]
        print(f"{r['model_name']:<40} {r['test_acc']:<12.4f} {r['test_auc']:<12.4f} "
              f"{r['test_f1']:<12.4f} {r['training_time']/60:<12.2f}")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comparison_results_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = value.copy()

    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(results, save_path=f"comparison_plot_{timestamp}.pdf")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Quantum Hydra models on PhysioNet EEG")

    # Model hyperparameters
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--qlcu-layers", type=int, default=2, help="QLCU circuit depth")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for classical model")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Data parameters
    parser.add_argument("--sample-size", type=int, default=10, help="Number of subjects")
    parser.add_argument("--sampling-freq", type=int, default=100, help="EEG sampling frequency")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    results = run_comparison(
        n_qubits=args.n_qubits,
        qlcu_layers=args.qlcu_layers,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        sampling_freq=args.sampling_freq,
        seed=args.seed,
        device=args.device
    )

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
