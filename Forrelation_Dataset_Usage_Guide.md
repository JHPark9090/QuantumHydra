# Usage Guide: Sequential Forrelation Dataset

This guide provides step-by-step instructions on how to generate and use the Sequential Forrelation dataset for training your sequence models.

## Overview

The data pipeline consists of two main files:

1.  **`generate_forrelation_dataset.py`**: A command-line script to create a custom dataset of forrelation sequences and save it to a `.pt` file.
2.  **`forrelation_dataloader.py`**: A Python module that provides a function to load the generated dataset into standard PyTorch `DataLoader` objects, ready for training.

---

## Step 1: Generate the Dataset

Before you can train your models, you must first generate the dataset file.

Open your terminal and run the `generate_forrelation_dataset.py` script. You can customize the dataset size and complexity using command-line arguments.

### Basic Usage

To create a default-sized dataset (2000 sequences, n_bits=6, seq_len=50):
```bash
python generate_forrelation_dataset.py
```

### Custom Usage

For a larger, more complex dataset, you can specify the arguments:
```bash
python generate_forrelation_dataset.py --num_pairs 5000 --n_bits 8 --seq_len 100 --filename large_forrelation_dataset.pt
```
*   `--num_pairs`: The total number of sequences (function pairs) to generate.
*   `--n_bits`: The number of input bits `n` for the Boolean functions. The domain size is `N = 2^n`. Higher values are exponentially more complex.
*   `--seq_len`: The number of samples `L` to include in each sequence.
*   `--filename`: The name of the output file.

This command will create a file (e.g., `large_forrelation_dataset.pt`) in your directory.

---

## Step 2: Use the DataLoader in Your Training Script

Once the dataset file is generated, you can easily load it for training.

In your main Python script for training, import the `get_forrelation_dataloader` function and use it to get your training and testing data loaders.

### Example Training Script

```python
import torch
import torch.nn as nn
from forrelation_dataloader import get_forrelation_dataloader
from TrueClassicalMamba import TrueClassicalMamba # Or any other model

# --- 1. Load the Data ---
# Specify the path to your generated dataset file
DATASET_FILE = "forrelation_dataset.pt"
BATCH_SIZE = 32

train_loader, test_loader, params = get_forrelation_dataloader(
    dataset_path=DATASET_FILE,
    batch_size=BATCH_SIZE
)

# --- 2. Instantiate Your Model ---
# The 'params' dictionary contains the necessary dimensions from the dataset
model = TrueClassicalMamba(
    n_channels=params['num_channels'],
    n_timesteps=params['seq_len'],
    output_dim=2  # Binary classification (High/Low Forrelation)
)

# --- 3. Standard Training Loop ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(5):
    for sequences, labels in train_loader:
        # Your training logic here...
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training finished.")
```

---

## Data Format and Model Compatibility

The dataloader is designed to be perfectly compatible with the models you have (`TrueClassicalMamba`, `TrueClassicalHydra`, etc.).

*   **Stored Shape:** The data is stored on disk as `(num_sequences, seq_len, num_channels)`.
*   **Model Input Shape:** The dataloader automatically transposes the data to `(batch_size, num_channels, seq_len)`, which is the shape your models expect.

The `params` dictionary returned by the dataloader is crucial for ensuring your model is instantiated with the correct dimensions:
```python
{'n_bits': 6, 'seq_len': 50, 'num_channels': 14}
```
This makes it easy to train different models on datasets of varying complexity without changing your code.
