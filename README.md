# HumanLSTM: LSTM vs BERT Transformer Comparison

## Overview

This project compares the performance of LSTM and BERT-like Transformer models on two behavioral prediction tasks: the Iterated Prisoner's Dilemma (IPD) and the Iowa Gambling Task (IGT). The codebase has evolved from a basic LSTM implementation (`main.py`) to an advanced comparison framework (`transformer.py`) that includes state-of-the-art transformer architecture.

## Key Improvements from main.py to transformer.py

### 🚀 **Major Architectural Enhancements**

#### 1. **BERT-like Transformer Implementation**
- **Added Complete Transformer Architecture**: Implemented a full BERT-like transformer with:
  - `PositionalEncoding`: Sinusoidal positional encodings for sequence awareness
  - `MultiHeadAttention`: Multi-head self-attention mechanism
  - `TransformerBlock`: Complete transformer blocks with residual connections
  - `bertModel`: End-to-end transformer model for sequence prediction

#### 2. **Enhanced Model Architecture**
- **Removed Softmax Activation**: Fixed regression task compatibility by removing softmax from both LSTM and BERT models
- **Improved Loss Functions**: Added robust MSE calculation with epsilon to prevent 0 MSE issues
- **Better Regularization**: Added weight decay, gradient clipping, and training noise

### 📊 **Advanced Training & Evaluation**

#### 3. **Comprehensive Model Comparison**
- **Multi-Model Evaluation**: Compare LSTM, BERT, AR (Autoregressive), and LR (Logistic Regression)
- **Robust MSE Calculation**: `robust_MSE_by_time()` function with edge case handling
- **Validation Monitoring**: Added validation loss tracking to detect overfitting
- **Debugging Information**: Detailed output analysis for model behavior

#### 4. **Training Improvements**
- **Reduced Learning Rate**: Changed from 1e-2 to 1e-3 for better convergence
- **Weight Decay**: Added L2 regularization (1e-4) to prevent overfitting
- **Gradient Clipping**: Prevents exploding gradients during training
- **Training Noise**: Small noise injection in early epochs to prevent memorization

### 🎯 **Task-Specific Enhancements**

#### 5. **IPD Task Improvements**
- **Enhanced Cooperation Rate Analysis**: Better visualization of cooperation patterns
- **Multi-Model Comparison Plots**: Side-by-side comparison of all models
- **Detailed Performance Metrics**: Comprehensive MSE and cooperation rate analysis

#### 6. **IGT Task Extensions**
- **BERT Model for IGT**: Extended transformer architecture to IGT task
- **Choice Rate Analysis**: Enhanced analysis of deck selection patterns
- **Correct Decision Tracking**: Better tracking of optimal decision-making

### 📈 **Visualization & Analysis**

#### 7. **Advanced Plotting**
- **Training vs Validation Loss**: Monitor overfitting during training
- **Model Comparison Plots**: Side-by-side performance comparisons
- **Cooperation Rate Analysis**: Detailed analysis of behavioral patterns
- **Loss Curve Analysis**: Training progress visualization

#### 8. **Performance Metrics**
- **Robust MSE Calculation**: Handles edge cases and prevents 0 MSE
- **Statistical Analysis**: Mean, standard deviation, and confidence intervals
- **Model Improvement Tracking**: Percentage improvements between models

## File Structure

```
HumanLSTM/
├── main.py              # Original LSTM-only implementation
├── transformer.py       # Enhanced LSTM + BERT comparison
├── data/
│   ├── IPD/            # Iterated Prisoner's Dilemma data
│   └── IGT/            # Iowa Gambling Task data
└── Figures/            # Generated plots and visualizations
```

## Key Features

### 🔧 **Technical Improvements**

1. **Transformer Architecture**
   ```python
   class bertModel(nn.Module):
       def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads=2, dropout=0.1):
           # Complete transformer implementation
   ```

2. **Robust Evaluation**
   ```python
   def robust_MSE_by_time(r, p, epsilon=1e-8):
       # Handles edge cases and prevents 0 MSE
   ```

3. **Advanced Training**
   ```python
   # Reduced learning rate, weight decay, gradient clipping
   optimizer_bert = optim.Adam(bert.parameters(), lr=1e-3, weight_decay=1e-4)
   torch.nn.utils.clip_grad_norm_(bert.parameters(), max_norm=1.0)
   ```

### 📊 **Analysis Capabilities**

1. **Multi-Model Comparison**: LSTM vs BERT vs AR vs LR
2. **Behavioral Pattern Analysis**: Cooperation rates, choice patterns
3. **Training Monitoring**: Validation loss, overfitting detection
4. **Statistical Analysis**: Confidence intervals, performance metrics

## Results & Performance

### 🎯 **Expected Improvements**

1. **Better Generalization**: BERT models show improved generalization over LSTM
2. **Robust Evaluation**: No more 0 MSE issues with proper regularization
3. **Comprehensive Analysis**: Detailed comparison across multiple models
4. **Behavioral Insights**: Better understanding of human decision patterns

### 📈 **Key Metrics**

- **MSE Comparison**: Robust comparison between LSTM and BERT performance
- **Cooperation Rates**: Analysis of cooperative behavior patterns
- **Training Stability**: Improved training with validation monitoring
- **Model Reliability**: Consistent performance across different tasks

## Usage

### Running the Enhanced Version
```bash
python transformer.py
```

### Key Outputs
- **Model Comparison Plots**: Performance comparison across all models
- **Training Curves**: Loss progression and validation monitoring
- **Behavioral Analysis**: Cooperation rates and choice patterns
- **Statistical Reports**: Detailed performance metrics

## Technical Details

### Model Architectures

#### LSTM Model
- **Architecture**: LSTM layers + ReLU + Linear output
- **Activation**: Removed softmax for regression compatibility
- **Training**: Standard backpropagation with Adam optimizer

#### BERT Model
- **Architecture**: Transformer blocks with multi-head attention
- **Components**: Positional encoding, self-attention, feed-forward networks
- **Training**: Advanced regularization and validation monitoring

### Data Processing
- **IPD Task**: 8258 trajectories, 9 time steps, binary cooperation decisions
- **IGT Task**: Multiple choice sequences, 4 deck options, 94 trials
- **Validation**: 5-fold cross-validation for robust evaluation

## Future Enhancements

1. **Additional Transformer Variants**: GPT-style, T5-style architectures
2. **Attention Visualization**: Understanding model decision patterns
3. **Real-time Analysis**: Interactive visualization tools
4. **Multi-task Learning**: Joint training on IPD and IGT tasks

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Citation

This work extends the original HumanLSTM framework with modern transformer architectures for behavioral prediction tasks. 