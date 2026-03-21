# CSI 5140 – Project 1: CIFAR-10 Image Classification

A from-scratch neural network for CIFAR-10 image classification built with manual implementations of core deep learning components (convolutions, optimizers, regularization) on top of PyTorch tensors.

## Project Structure

```
Project1/
├── cifar10.ipynb          # Main notebook — all implementation
├── pyproject.toml         # Python project config & dependencies
├── uv.lock                # Locked dependency versions (uv package manager)
├── README.md
├── data/
│   ├── cifar-10-batches-py/   # Extracted CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-python.tar.gz # Raw dataset archive
├── model_a.pth            # Saved model checkpoint (Option A)
├── model_b.pth            # Saved model checkpoint (Option B)
├── model_c.pth            # Saved model checkpoint (Option C)
└── ModelArch.drawio        # Model architecture diagram (editable)
```

## Notebook Overview (`cifar10.ipynb`)

The notebook is organized into the following sections:

| Section | Description |
|---|---|
| **Activation Layers** | Manual ReLU, Softmax, and Sigmoid implementations |
| **Loss Function** | Cross-entropy loss and L2 regularization |
| **Optimizers** | SGD, Momentum, RMSprop, and Adam |
| **Learning Rate Decay** | Configurable decay rates |
| **Convolutional Layer** | `conv_forward` implementation |
| **Data Loading** | CIFAR-10 download & augmentation via `torchvision` |
| **Model Parameters** | Architecture definition and hyperparameter setup |
| **Training Loop** | `train_model()` function with configurable optimizer, regularization, and decay |
| **Ablation Study** | Comparisons of optimizers, L2 λ values, dropout vs. L2, Adam β tuning, and LR decay |
| **Final Models** | Three best configurations trained and saved as `.pth` checkpoints |

## Requirements

- Python ≥ 3.10
- Dependencies (managed via [uv](https://github.com/astral-sh/uv)):
  - `torch ≥ 2.10.0`
  - `torchvision ≥ 0.25.0`
  - `numpy ≥ 2.2.6`
  - `matplotlib ≥ 3.10.8`
  - `ipykernel ≥ 7.2.0`

## How to Run

### 1. Install dependencies

```bash
# Install uv if you don't have it
pip install uv

# Create the virtual environment and install all dependencies
uv sync
```

### 2. Launch the notebook

```bash
# Activate the virtual environment (macOS)
source .venv/bin/activate

# Start Jupyter
jupyter notebook cifar10.ipynb
```

Alternatively, open `cifar10.ipynb` directly in VS Code with the Jupyter extension (select the `.venv` kernel).

### 3. Run all cells

Execute the cells sequentially from top to bottom. The notebook will:

1. Download CIFAR-10 to `data/` (first run only)
2. Define all layers, loss functions, and optimizers
3. Train the model and run the ablation study
4. Save the three best model checkpoints as `model_a.pth`, `model_b.pth`, `model_c.pth`

> **Hint:** The notebook can be run form top to bottom without any modifications.

> **Note:** Training uses GPU acceleration when available (MPS on Apple Silicon, CUDA on NVIDIA). CPU fallback is automatic.