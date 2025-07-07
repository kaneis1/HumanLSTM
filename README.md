# HumanLSTM: Advanced Behavioral Prediction with LSTM and Transformer Models

## 📋 Overview

This project implements and compares advanced neural network architectures for behavioral prediction tasks, specifically the **Iterated Prisoner's Dilemma (IPD)** and **Iowa Gambling Task (IGT)**. The codebase has evolved from basic LSTM implementations to sophisticated transformer-based models with multiple versions offering different approaches to the prediction problem.

## 🚀 Project Evolution

### Version Progression
- **`main.py`**: Original LSTM implementation
- **`ipd_bert.py`**: Basic LSTM vs BERT comparison
- **`ipd_bert_v2.py`**: Enhanced transformer with improved training
- **`ipd_bert_v3.py`**: Single prediction using 8 moves to predict 9th move
- **`ipd_bert_v4.py`**: Advanced BERT with sequence + feature fusion
- **`igt_bert.py`**: Iowa Gambling Task with BERT transformer

## 🏗️ Architecture Overview

### Core Components

#### 1. **Transformer Architecture**
```python
class bertModel(nn.Module):
    - PositionalEncoding: Sinusoidal positional encodings
    - MultiHeadAttention: Multi-head self-attention mechanism
    - TransformerBlock: Complete transformer blocks with residuals
    - Input/Output layers with proper activation functions
```

#### 2. **LSTM Architecture**
```python
class lstmModel(nn.Module):
    - LSTM layers with configurable hidden dimensions
    - ReLU activation and dropout for regularization
    - Softmax output for classification tasks
```

#### 3. **Advanced BERT v4 Architecture**
```python
class bertModel_v4(nn.Module):
    - Dual input: sequence data + engineered features
    - Feature fusion: combines sequence and regression features
    - Global pooling: mean pooling over sequence dimension
    - Enhanced attention with feature context
```

## 📊 Model Versions Comparison

### IPD Task Versions

| Version | Description | Key Features | Prediction Target |
|---------|-------------|--------------|-------------------|
| `ipd_bert.py` | Basic comparison | LSTM vs BERT | Multiple future moves |
| `ipd_bert_v2.py` | Enhanced training | Improved loss, validation | Multiple future moves |
| `ipd_bert_v3.py` | Single prediction | 8 moves → 9th move | Single move prediction |
| `ipd_bert_v4.py` | Feature fusion | Sequence + regression features | Single move prediction |

### IGT Task
| Version | Description | Key Features | Prediction Target |
|---------|-------------|--------------|-------------------|
| `igt_bert.py` | Iowa Gambling Task | 4-deck choice prediction | Next deck choice |

## 🔧 Technical Features

### Enhanced Training Features
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Regularization**: Weight decay, dropout, gradient clipping
- **Validation Monitoring**: Real-time validation loss tracking
- **Robust MSE**: Enhanced MSE calculation with epsilon handling
- **Learning Rate Scheduling**: Adaptive learning rates for better convergence

### Data Processing
- **IPD Data**: 8258 games with 9 moves each
- **IGT Data**: Multiple choice sequences with 4 decks
- **Feature Engineering**: Regression features for v4
- **Sequence Preparation**: Sliding windows and padding

### Visualization System
- **Comprehensive Plots**: MSE, cooperation rates, training loss
- **Statistical Analysis**: Error bars and confidence intervals
- **Model Comparisons**: Side-by-side performance analysis
- **Training Monitoring**: Loss curves and validation tracking

## 📁 File Structure

```
HumanLSTM/
├── main.py                    # Original LSTM implementation
├── ipd_bert.py               # Basic LSTM vs BERT comparison
├── ipd_bert_v2.py            # Enhanced transformer (lag-based)
├── ipd_bert_v3.py            # Single prediction (8→9 moves)
├── ipd_bert_v4.py            # Feature fusion BERT
├── igt_bert.py               # Iowa Gambling Task BERT
├── README.md                 # This documentation
├── data/
│   ├── IPD/                  # IPD task data
│   │   ├── all_data.csv      # Main IPD dataset
│   │   └── processed_*.pkl   # Processed data files
│   └── IGT/                  # IGT task data
│       ├── choice_*.csv      # IGT choice data
│       └── processed_*.pkl   # Processed IGT data
├── Figures/
│   ├── IPD_lag_4/           # IPD v2 plots
│   ├── IPD_v3/              # IPD v3 plots
│   ├── IPD_v4/              # IPD v4 plots
│   └── IGT/                 # IGT plots
└── bert_model_*.pth         # Saved model weights
```

## 🎯 Key Innovations

### Version-Specific Features

#### **v2 Enhancements**
- Fixed 0 MSE issue with robust loss calculation
- Added validation monitoring and early stopping
- Improved regularization with weight decay
- Enhanced visualization with statistical analysis

#### **v3 Innovations**
- Single prediction task: 8 moves → 9th move
- Simplified architecture focused on one-step prediction
- Direct comparison of cooperation probability
- Streamlined evaluation metrics

#### **v4 Advanced Features**
- **Dual Input Architecture**: Sequence + engineered features
- **Feature Fusion**: Combines temporal and regression features
- **Enhanced Attention**: Attends to both sequence and feature patterns
- **Rich Feature Set**: 26+ engineered features including interactions

#### **IGT Specialization**
- **4-Deck Prediction**: Specialized for Iowa Gambling Task
- **Choice Sequence**: Predicts next deck choice from history
- **Performance**: 88.95% improvement over LSTM baseline

## 📈 Performance Results

### IPD Task Performance (MSE)
| Model | v2 | v3 | v4 |
|-------|----|----|----|
| BERT | ~0.122 | ~0.152 | 0.114 |
| LSTM | ~0.211 | ~0.150 | ~0.150 |
| AR | ~0.160 | ~0.246 | ~0.246 |
| LR | ~0.750 | ~0.782 | ~0.782 |

### IGT Task Performance
- **LSTM MSE**: 0.144642
- **BERT MSE**: 0.015981
- **Improvement**: 88.95% over LSTM

## 🛠️ Usage Instructions

### Running Different Versions

```bash
# Basic LSTM vs BERT comparison
python ipd_bert.py

# Enhanced transformer with lag-based prediction
python ipd_bert_v2.py

# Single prediction (8 moves → 9th move)
python ipd_bert_v3.py

# Advanced feature fusion BERT
python ipd_bert_v4.py

# Iowa Gambling Task
python igt_bert.py
```

### Key Parameters
```python
# Model Architecture
n_nodes = 10        # Hidden dimension
n_layers = 2        # Number of layers
num_heads = 2       # Attention heads (BERT)

# Training Parameters
n_epochs = 10       # Training epochs
batch_size = 100    # Batch size
learning_rate = 1e-4 # Learning rate (BERT)
weight_decay = 1e-4 # L2 regularization

# Task-Specific
input_length = 8    # Input sequence length (v3/v4)
lag = 7            # Lag for prediction (v2)
```

## 🔍 Model Architectures Deep Dive

### BERT v4 Feature Fusion
```python
# Input Processing
seq_data: (batch_size, 8, 2)      # 8 moves from 2 players
feature_data: (batch_size, 26)    # Engineered regression features

# Feature Fusion
seq_embedded = seq_embedding(seq_data)
feature_embedded = feature_embedding(feature_data)
combined = seq_embedded + feature_embedded

# Transformer Processing
for transformer_block in transformer_blocks:
    combined = transformer_block(combined)

# Output
pooled = torch.mean(combined, dim=1)
output = output_layer(pooled)
```

### Enhanced Training Features
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rates
- **Early Stopping**: Based on validation loss
- **Data Augmentation**: Small noise for regularization

## 📊 Visualization Features

### Generated Plots
1. **Cooperation Rate Comparison**: All models vs human behavior
2. **MSE Comparison**: Performance metrics across models
3. **Training Loss Curves**: LSTM vs BERT training progress
4. **Validation Loss**: Overfitting detection
5. **Model-Specific Plots**: Individual model performance
6. **Statistical Analysis**: Error bars and confidence intervals

### Plot Organization
- **IPD v2**: `Figures/IPD_lag_4/`
- **IPD v3**: `Figures/IPD_v3/`
- **IPD v4**: `Figures/IPD_v4/`
- **IGT**: `Figures/IGT/`

## 🐛 Problem Resolution

### 0 MSE Issue (Fixed in v2)
**Root Cause**: Model overfitting with perfect memorization
**Solutions Applied**:
1. Removed inappropriate softmax activation
2. Added robust MSE calculation with epsilon
3. Implemented proper regularization
4. Added validation monitoring
5. Reduced learning rate and added weight decay

### Training Stability Improvements
- **Gradient Clipping**: Prevents gradient explosion
- **Weight Decay**: L2 regularization for overfitting
- **Validation Monitoring**: Real-time overfitting detection
- **Learning Rate Reduction**: Better convergence

## 📚 Dependencies

```python
# Core ML Libraries
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Statistical Analysis
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.13.0

# Data Processing
pickle
```

## 🔮 Future Enhancements

### Planned Improvements
1. **Hyperparameter Optimization**: Grid search and Bayesian optimization
2. **Attention Visualization**: Interpretable attention weights
3. **Multi-Task Learning**: Joint IPD and IGT training
4. **Ensemble Methods**: Model combination for improved performance
5. **Real-time Interface**: Web-based prediction interface
6. **Attention Analysis**: Understanding model decision-making
7. **Cross-Dataset Validation**: Testing on additional behavioral datasets

### Research Directions
- **Interpretability**: Understanding model predictions
- **Transfer Learning**: Pre-trained models for new tasks
- **Multi-Modal Fusion**: Combining different data types
- **Real-time Adaptation**: Online learning capabilities

## 📖 Citation

This work extends the original HumanLSTM framework with modern transformer architectures for behavioral prediction tasks. The project demonstrates the effectiveness of attention-based models in capturing complex behavioral patterns in game-theoretic scenarios.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the codebase.

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project represents a comprehensive comparison of traditional LSTM and modern transformer architectures for behavioral prediction, with multiple versions offering different approaches to the same fundamental problem of predicting human behavior in strategic interactions.