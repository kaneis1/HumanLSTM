# HumanLSTM: LSTM vs BERT Transformer Comparison

## Overview

This project compares the performance of LSTM and BERT-like Transformer models on behavioral prediction tasks: the Iterated Prisoner's Dilemma (IPD). The codebase has evolved from a basic LSTM implementation (`main.py`) to an advanced comparison framework (`ipd_bert.py`) that includes state-of-the-art transformer architecture, and now includes a dedicated IGT BERT training module (`igt_bert.py`).

## Key Improvements from main.py to ipd_bert.py

### 🚀 **Major Architectural Enhancements**

#### 1. **BERT-like Transformer Implementation**
- **Added Complete Transformer Architecture**: Implemented a full BERT-like transformer with:
  - `PositionalEncoding`: Sinusoidal positional encodings for sequence awareness
  - `MultiHeadAttention`: Multi-head self-attention mechanism
  - `TransformerBlock`: Complete transformer blocks with residual connections
  - `bertModel`: Full BERT-like model for IPD task prediction

#### 2. **Enhanced Model Architecture**
- **Removed Softmax Activation**: Fixed regression task compatibility by removing softmax from output layers
- **Robust MSE Calculation**: Added `robust_MSE_by_time()` function to prevent 0 MSE issues
- **Improved Loss Functions**: Better loss calculation with epsilon to handle edge cases
- **Regularization**: Added weight decay and gradient clipping to prevent overfitting

### 📊 **Advanced Training & Evaluation**

#### 3. **Comprehensive Model Comparison**
- **Multi-Model Evaluation**: Compares LSTM, BERT, AR (Autoregressive), and LR (Logistic Regression)
- **Cross-Validation**: 5-fold cross-validation for robust performance assessment
- **Robust MSE Metrics**: Enhanced MSE calculation that handles edge cases and prevents 0 MSE

#### 4. **Training Improvements**
- **Reduced Learning Rate**: Changed BERT learning rate from 1e-2 to 1e-3 for better convergence
- **Weight Decay**: Added L2 regularization (1e-4) to prevent overfitting
- **Gradient Clipping**: Prevents exploding gradients during training
- **Training Noise**: Small noise added early in training to prevent perfect memorization
- **Validation Monitoring**: Tracks validation loss to detect overfitting

### 🎨 **Enhanced Visualization**

#### 5. **Comprehensive Plotting System**
- **Multiple Comparison Plots**: MSE comparison, cooperation rate prediction, training loss
- **Statistical Analysis**: Error bars and confidence intervals for all predictions
- **Model-Specific Plots**: Individual plots for each model (LSTM, BERT, AR, LR)
- **Training Monitoring**: Loss curves and validation loss tracking

## New Addition: IGT BERT Training (`igt_bert.py`)

### 🆕 **IGT Task with BERT Transformer**

#### 6. **Dedicated IGT Module**
- **Separate Implementation**: Created `igt_bert.py` specifically for IGT task with BERT
- **IGT-Specific Models**: `lstmIGT` and `bertIGT` models adapted for 4-deck choice prediction
- **Data Processing**: Loads IGT data from CSV files and converts to one-hot encoding
- **Sequence Preparation**: Creates sliding window sequences for training

#### 7. **IGT Training Features**
- **Train/Test Split**: 80/20 split for proper evaluation
- **Sequence Length**: Configurable sequence length (default: 10 trials)
- **Batch Processing**: Efficient batch training with configurable batch size
- **Validation**: Real-time validation loss monitoring

#### 8. **IGT Results**
- **Performance**: BERT achieved 88.95% improvement over LSTM on IGT task
- **MSE Comparison**: 
  - LSTM MSE: 0.144642
  - BERT MSE: 0.015981
- **Separate Figure Folder**: All IGT plots saved to `Figures/IGT/`

### 📁 **File Structure**

```
HumanLSTM/
├── main.py                 # Original LSTM implementation
├── ipd_bert.py            # Enhanced LSTM + BERT for IPD task
├── igt_bert.py            # BERT training for IGT task
├── README.md              # This documentation
├── data/
│   ├── IPD/              # IPD task data
│   └── IGT/              # IGT task data
└── Figures/
    ├── *.png             # IPD task plots
    └── IGT/              # IGT task plots
        ├── igt_model_comparison_*.png
        ├── igt_training_loss_*.png
        ├── igt_bert_val_loss_*.png
        └── igt_mse_comparison_*.png
```

## Model Architectures

### IPD Task Models (`ipd_bert.py`)
- **LSTM Model**: Traditional LSTM with ReLU activation
- **BERT Model**: Full transformer architecture with positional encoding and multi-head attention
- **AR Model**: Autoregressive model for baseline comparison
- **LR Model**: Logistic regression for baseline comparison

### IGT Task Models (`igt_bert.py`)
- **LSTM IGT**: LSTM adapted for 4-deck choice prediction
- **BERT IGT**: Transformer adapted for IGT sequence prediction

## Results Summary

### IPD Task Performance
- **LSTM MSE**: ~0.116
- **BERT MSE**: Significantly improved (with fixes applied)
- **AR MSE**: ~0.183
- **LR MSE**: ~0.750

### IGT Task Performance
- **LSTM MSE**: 0.144642
- **BERT MSE**: 0.015981
- **BERT Improvement**: 88.95% over LSTM

## Problem Resolution: 0 MSE Issue

### 🔧 **Root Cause Analysis**
The BERT model initially showed almost 0 MSE due to:
1. **Overfitting**: Model memorized training data perfectly
2. **Softmax with MSE**: Inappropriate activation function for regression
3. **No Regularization**: Lack of proper regularization techniques
4. **Autoencoder-like Training**: Same data used for input and target

### ✅ **Applied Fixes**
1. **Model Architecture**: Removed softmax activation from output layers
2. **Training Improvements**: Reduced learning rate, added weight decay, gradient clipping
3. **Robust Evaluation**: Added epsilon to MSE calculation to prevent 0 values
4. **Validation Monitoring**: Added validation loss tracking to detect overfitting
5. **Training Noise**: Small noise added early in training to prevent memorization

## Usage Instructions

### Running IPD Task
```bash
python ipd_bert.py
```

### Running IGT Task
```bash
python igt_bert.py
```

### Key Parameters
- **n_nodes**: Hidden dimension (default: 10)
- **n_layers**: Number of layers (default: 2)
- **n_epochs**: Training epochs (default: 10 for IPD, 20 for IGT)
- **batch_size**: Batch size for training (default: 100 for IPD, 32 for IGT)

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Future Enhancements

1. **Hyperparameter Optimization**: Grid search for optimal model parameters
2. **Attention Visualization**: Plot attention weights for interpretability
3. **Multi-Task Learning**: Joint training on IPD and IGT tasks
4. **Ensemble Methods**: Combine multiple models for improved performance
5. **Real-time Prediction**: Web interface for real-time behavioral prediction

## Citation

This work extends the original HumanLSTM framework with modern transformer architectures for behavioral prediction tasks. 