import torch
import torch.nn as nn
import os
import sys

# --- Import Dataset Scripts ---
from generate_forrelation_dataset import generate_dataset
from forrelation_dataloader import get_forrelation_dataloader

# --- Import All Model Architectures ---
# Add current directory to path to ensure modules are found
sys.path.append(os.getcwd())

# It's good practice to wrap imports in try/except blocks for clear error messages
try:
    from TrueClassicalMamba import TrueClassicalMamba
    print("Successfully imported TrueClassicalMamba")
except ImportError:
    print("Error: Could not import TrueClassicalMamba. Make sure TrueClassicalMamba.py exists.")
    TrueClassicalMamba = None

try:
    from TrueClassicalHydra import TrueClassicalHydra
    print("Successfully imported TrueClassicalHydra")
except ImportError:
    print("Error: Could not import TrueClassicalHydra. Make sure TrueClassicalHydra.py exists.")
    TrueClassicalHydra = None

# NOTE: The following are placeholders.
# Replace them with your actual Quantum model imports when they are ready.
# For this test, we will make them aliases of the classical versions
# to confirm the data pipeline works for a model with the *same structure*.
try:
    from QuantumMamba import QuantumMamba
    print("Successfully imported QuantumMamba")
except ImportError:
    print("Warning: QuantumMamba.py not found. Using a placeholder for compatibility testing.")
    QuantumMamba = TrueClassicalMamba

try:
    from QuantumMambaHybrid import QuantumMambaHybrid
    print("Successfully imported QuantumMambaHybrid")
except ImportError:
    print("Warning: QuantumMambaHybrid.py not found. Using a placeholder for compatibility testing.")
    QuantumMambaHybrid = TrueClassicalMamba

try:
    from QuantumHydra import QuantumHydra
    print("Successfully imported QuantumHydra")
except ImportError:
    print("Warning: QuantumHydra.py not found. Using a placeholder for compatibility testing.")
    QuantumHydra = TrueClassicalMamba

try:
    from QuantumHydraHybrid import QuantumHydraHybrid
    print("Successfully imported QuantumHydraHybrid")
except ImportError:
    print("Warning: QuantumHydraHybrid.py not found. Using a placeholder for compatibility testing.")
    QuantumHydraHybrid = TrueClassicalMamba


def test_model_training_step(model_class, model_name, dataset_params):
    """
    Tests a single training step for a given model class.
    """
    print("-" * 60)
    print(f"Testing model: {model_name}")
    
    if model_class is None:
        print("Result: ❌ FAILED (Model could not be imported)")
        return

    try:
        # 1. Instantiate the model using dataset parameters
        model = model_class(
            n_channels=dataset_params['num_channels'],
            n_timesteps=dataset_params['seq_len'],
            output_dim=2  # Binary classification
        )
        print("  - Model instantiated successfully.")

        # 2. Get a batch of data
        train_loader, _, _ = get_forrelation_dataloader(dataset_path=TEST_DATASET_FILENAME, batch_size=4)
        sequences, labels = next(iter(train_loader))
        print(f"  - Data batch loaded successfully with shape: {sequences.shape}")

        # 3. Perform a forward pass
        outputs = model(sequences)
        print(f"  - Forward pass successful. Output shape: {outputs.shape}")

        # 4. Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f"  - Loss calculated: {loss.item():.4f}")

        # 5. Perform backward pass and optimizer step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("  - Backward pass and optimizer step successful.")
        
        print("Result: ✅ PASSED")

    except Exception as e:
        print(f"  - An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
        print(f"Result: ❌ FAILED")


if __name__ == "__main__":
    TEST_DATASET_FILENAME = "temp_forrelation_test_dataset.pt"
    
    # 1. Generate a small temporary dataset for the test
    print("=" * 60)
    print("Step 1: Generating a small temporary dataset for testing...")
    generate_dataset(
        num_pairs=50,
        n_bits=4,
        seq_len=32,
        filename=TEST_DATASET_FILENAME
    )
    print("=" * 60)
    
    # 2. Load the dataset to get its parameters
    print("\nStep 2: Loading dataset parameters...")
    _, _, dataset_params = get_forrelation_dataloader(dataset_path=TEST_DATASET_FILENAME)
    print("=" * 60)

    # 3. Run the compatibility test for each model
    print("\nStep 3: Running compatibility tests...")
    models_to_test = {
        "TrueClassicalMamba": TrueClassicalMamba,
        "TrueClassicalHydra": TrueClassicalHydra,
        "QuantumMamba": QuantumMamba,
        "QuantumMambaHybrid": QuantumMambaHybrid,
        "QuantumHydra": QuantumHydra,
        "QuantumHydraHybrid": QuantumHydraHybrid,
    }

    for name, model_class in models_to_test.items():
        test_model_training_step(model_class, name, dataset_params)
        
    # 4. Clean up the temporary dataset file
    os.remove(TEST_DATASET_FILENAME)
    print("-" * 60)
    print(f"Temporary dataset file '{TEST_DATASET_FILENAME}' has been removed.")
    print("=" * 60)
    print("Compatibility test finished.")
    print("=" * 60)
