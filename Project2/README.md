# CSI 5140 – Project 2: CIFAR-10 Image Classification Model Compression

A from-scratch neural network for CIFAR-10 image classification built with manual implementations of core deep learning components (convolutions, optimizers, regularization) on top of PyTorch tensors.
Then the model is compressed using pruning and quantization.

## Project Structure

```
Project2/
├── cifar10.ipynb          # Main notebook — all implementation
├── pyproject.toml         # Python project config & dependencies
├── uv.lock                # Locked dependency versions (uv package manager)
├── README.md
├── data/
│   ├── cifar-10-batches-py/   # Extracted CIFAR-10 dataset(auto-downloaded)
│   └── cifar-10-python.tar.gz # Raw dataset archive
├── model_c.pth                # Best saved model from Project 1
├── model_pruned.pth           # Pruned only model
├── model_bit.pth              # Quantized only model
└── model_combined.pth         # Pruned + Quantized model
```

## Notebook Overview (`cifar10.ipynb`)

The notebook is organized into the following sections:

| Section | Description |
|---|---|
| **Implementation - Mainly from Project 1 taken** | Manual implementation from Project 1 cleaned up for Project 2 and enhanved with `forward_pass()` and `eval_model_accuracy()` |
| **Baseline Model** | Load model c as baseline model for Project 2, `model_report()` is used to evaluate each model |
| **Magnitude Based Pruning** | Flatten weights and set smallest fractions of weights to 0. Then retrain the model for fine tuning. |
| **Quantization** | Quantize using Kmeans to x-bit integers. Then retrain the model for fine tuning. |
| **Training Loop** | `deep_Compression()` function with configurable pruning and quantization |
| **Ablation Study** | Comparisons of different pruning and quantization configurations and the effect of fine tuning. |
| **Evaluation & Plotting** | Compare the baseline model with the pruned and quantized models |
| **Inference Evaluation** | Evaluate the pruned and quantized models on the test set on CPU, CUDA and MPS and compare the inference time and accuracy |

## Requirements

- Python ≥ 3.10
- Dependencies (managed via [uv](https://github.com/astral-sh/uv)):
  - `torch ≥ 2.10.0`
  - `torchvision ≥ 0.25.0`
  - `numpy ≥ 2.2.6`
  - `matplotlib ≥ 3.10.8`
  - `ipykernel ≥ 7.2.0`
  - `scikit-learn ≥ 1.7.2`

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
4. Save the three different models as checkpoints as `model_pruned.pth`, `model_bit.pth`, `model_combined.pth`

> **Hint:** The notebook can be run form top to bottom without any modifications, but it is recommended to skip the model_c regeneration and just use the provided `model_c.pth` checkpoint.

> **Note:** Training uses GPU acceleration when available (MPS on Apple Silicon, CUDA on NVIDIA). CPU fallback is automatic. (Except for inference on Edge devices)