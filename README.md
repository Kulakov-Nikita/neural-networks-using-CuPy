# Neural Networks Using CuPy - Course Project

This repository contains five implementations of different neural network architectures for various tasks, using CuPy for GPU acceleration. Each file corresponds to a separate laboratory assignment.

## 🛠️ Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (required for CuPy)
- **NVIDIA CUDA Toolkit 11.x/12.x/13.x** installed

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Important Note about CuPy**: The `requirements.txt` specifies `cupy-cuda13x`. You may need to install a different version matching your CUDA installation:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

Check your CUDA version with:
```bash
nvidia-smi
```

## 🚀 Running the Experiments

### Convolutional Neural Network (MNIST)
```bash
python main_conv.py
```
**Outputs:**
- Training loss plot
- Accuracy, F1, and ROC AUC scores
- ROC curves for each class

### Dense Network (Mushrooms)
```bash
python main_dense.py
```
**Outputs:**
- Training loss plot
- Accuracy, F1, precision, recall metrics
- ROC curve

### Recurrent Networks (Steel Industry)
```bash
python main_recurrent.py
```
**Outputs:**
- Comparative training loss plot (RNN, LSTM, GRU)
- Accuracy, F1, MSE, RMSE, R² scores for each architecture

### VAE + GAN (MNIST)
```bash
code main_vae_gan.ipynb
```

### Transformer (Toxic Comments)
```bash
python main_transformer.py
```
**Outputs:**
- Training loss plot
- Accuracy and F1 scores
- Note: Requires dataset file at specified location

## 🧪 Key Features

- **GPU Acceleration**: All models leverage CuPy for GPU computation
- **Modular Architecture**: Clean separation between layers, models, and utilities
- **Multiple Architectures**: CNN, Dense, RNN/LSTM/GRU, and Transformer implementations
- **Comprehensive Metrics**: Accuracy, F1, precision, recall, ROC, AUC, MSE, RMSE, R²

## ⚠️ Troubleshooting

### CuPy Installation Issues
If you encounter CuPy installation problems:
1. Ensure your CUDA version matches the CuPy version
2. Try installing from conda: `conda install -c conda-forge cupy`
3. Or build from source: [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html)

### Memory Errors
- Reduce batch sizes in the code if you encounter GPU memory issues
- Adjust `load_percent` parameter in `main_*.py` for smaller dataset

### Missing Data Files
- Check that all data loaders are properly implemented
- Ensure dataset files are in the correct locations

## 📈 Expected Results

Each script will:
1. Train the specified neural network architecture
2. Display training progress (loss over epochs)
3. Evaluate on test data with multiple metrics
4. Generate visualization plots

## 🛠️ Customization

You can modify:
- Hyperparameters (learning rate, batch size, epochs)
- Network architectures (layer sizes, activation functions)
- Dataset loading parameters
- Evaluation metrics

## 📚 Dependencies

- **CuPy**: GPU-accelerated computing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **scikit-learn**: Metrics and utilities
- **TensorFlow/Keras**: Loading MNIST dataset
