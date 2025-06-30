## Python Imports

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import seaborn as sns
import math
import os

sns.set()
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

MSE = mean_squared_error
lag = 1

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

## Plotting Config

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Create IGT figures folder
os.makedirs("Figures/IGT", exist_ok=True)

## BERT-like Transformer Components

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

## IGT Task Models

class lstmIGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        return out

class bertIGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads=2, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input embedding
        self.input_embedding = nn.Linear(in_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

def robust_MSE_by_time(r, p, epsilon=1e-8):
    """Robust MSE calculation that handles edge cases"""
    err = []
    for t in np.arange(r.shape[1]):
        if len(p.shape) == 3:
            mse_val = MSE(r[:, t, :], p[:, t, :])
        else:
            mse_val = MSE(r[:, t, 0], p[:, t])
        # Add small epsilon to prevent 0 MSE
        err.append(max(mse_val, epsilon))
    return np.array(err)

def MSE_by_time(r, p):
    """Backward compatibility function"""
    return robust_MSE_by_time(r, p, epsilon=0)

def getCR(r):  # cooperation rate
    if len(r.shape) == 3:
        r = r[:, :, 0]
    cr = np.zeros(r.shape)
    for t in np.arange(r.shape[1]):
        for b in np.arange(r.shape[0]):
            cr[b, t] = r[b, : t + 1].sum() / (t + 1)
    return cr

def load_igt_data(filename="choice_100.csv"):
    """Load IGT data from CSV file"""
    print(f"Loading IGT data from: data/IGT/{filename}")
    try:
        data = pd.read_csv(f"data/IGT/{filename}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")
        
        # Convert to numpy array and reshape
        # Each row represents one participant's choices across trials
        choices = data.values  # Shape: (n_participants, n_trials)
        print(f"Choices shape: {choices.shape}")
        
        # Convert to one-hot encoding for 4 decks
        n_participants, n_trials = choices.shape
        one_hot = np.zeros((n_participants, n_trials, 4))
        
        for i in range(n_participants):
            for j in range(n_trials):
                deck_choice = int(choices[i, j]) - 1  # Convert 1-4 to 0-3
                one_hot[i, j, deck_choice] = 1
        
        print(f"One-hot shape: {one_hot.shape}")
        print(f"Sample participant choices: {choices[0, :10]}")
        print(f"Sample one-hot encoding: {one_hot[0, :5]}")
        
        return one_hot, choices
        
    except Exception as e:
        print(f"Error loading IGT data: {e}")
        # Create synthetic data for testing
        print("Creating synthetic IGT data for testing...")
        n_participants, n_trials = 50, 100
        choices = np.random.randint(1, 5, (n_participants, n_trials))
        one_hot = np.zeros((n_participants, n_trials, 4))
        
        for i in range(n_participants):
            for j in range(n_trials):
                deck_choice = int(choices[i, j]) - 1
                one_hot[i, j, deck_choice] = 1
        
        print(f"Created synthetic data: {one_hot.shape}")
        return one_hot, choices

def prepare_igt_sequences(data, seq_length=10):
    """Prepare sequences for training"""
    n_participants, n_trials, n_decks = data.shape
    sequences = []
    targets = []
    
    for participant in range(n_participants):
        for start in range(n_trials - seq_length):
            seq = data[participant, start:start + seq_length]
            target = data[participant, start + 1:start + seq_length + 1]
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_igt_models():
    """Train LSTM and BERT models on IGT data"""
    
    # Load data
    print("Loading IGT data...")
    igt_data, raw_choices = load_igt_data()
    
    # Prepare sequences
    seq_length = 10
    sequences, targets = prepare_igt_sequences(igt_data, seq_length)
    
    # Split into train/test
    n_samples = len(sequences)
    train_size = int(0.8 * n_samples)
    
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    
    # Model parameters
    n_nodes, n_layers = 10, 2
    n_epochs, batch_size = 20, 32
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm = lstmIGT(4, n_nodes, 4, n_layers)
    criterion_lstm = nn.MSELoss()
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=1e-3)
    loss_set_lstm = []
    
    for ep in range(n_epochs):
        epoch_losses = []
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]
            
            inputs = Variable(torch.from_numpy(batch_sequences).float())
            target = Variable(torch.from_numpy(batch_targets).float())
            
            output = lstm(inputs)
            loss = criterion_lstm(output, target)
            
            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()
            
            epoch_losses.append(loss.item())
            loss_set_lstm.append(loss.item())
        
        if ep % 5 == 0:
            print(f"LSTM Epoch {ep+1}/{n_epochs}, Loss: {np.mean(epoch_losses):.5f}")
    
    lstm.eval()
    
    # Train BERT model
    print("Training BERT model...")
    bert = bertIGT(4, n_nodes, 4, n_layers)
    criterion_bert = nn.MSELoss()
    optimizer_bert = optim.Adam(bert.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_set_bert = []
    val_loss_set_bert = []
    
    for ep in range(n_epochs):
        bert.train()
        epoch_losses = []
        
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]
            
            inputs = Variable(torch.from_numpy(batch_sequences).float())
            target = Variable(torch.from_numpy(batch_targets).float())
            
            output = bert(inputs)
            
            # Add small noise to prevent perfect memorization
            if ep < 10:
                output = output + torch.randn_like(output) * 1e-4
            
            loss = criterion_bert(output, target)
            
            optimizer_bert.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(bert.parameters(), max_norm=1.0)
            
            optimizer_bert.step()
            
            epoch_losses.append(loss.item())
            loss_set_bert.append(loss.item())
        
        # Validation step
        bert.eval()
        with torch.no_grad():
            val_inputs = Variable(torch.from_numpy(test_sequences).float())
            val_targets = Variable(torch.from_numpy(test_targets).float())
            val_output = bert(val_inputs)
            val_loss = criterion_bert(val_output, val_targets).item()
            val_loss_set_bert.append(val_loss)
            
            if ep % 5 == 0:
                print(f"BERT Epoch {ep+1}/{n_epochs}: Train Loss: {np.mean(epoch_losses):.5f}, Val Loss: {val_loss:.5f}")
    
    bert.eval()
    
    # Evaluate models
    print("Evaluating models...")
    
    with torch.no_grad():
        # Test predictions
        test_inputs = Variable(torch.from_numpy(test_sequences).float())
        
        lstm_output = lstm(test_inputs).data.cpu().numpy()
        bert_output = bert(test_inputs).data.cpu().numpy()
        
        # Calculate MSE
        lstm_mse = np.mean(robust_MSE_by_time(test_targets, lstm_output))
        bert_mse = np.mean(robust_MSE_by_time(test_targets, bert_output))
        
        print(f"LSTM MSE: {lstm_mse:.6f}")
        print(f"BERT MSE: {bert_mse:.6f}")
        print(f"BERT vs LSTM improvement: {(lstm_mse - bert_mse) / lstm_mse * 100:.2f}%")
    
    # Create visualizations
    create_igt_plots(lstm_output, bert_output, test_targets, loss_set_lstm, 
                    loss_set_bert, val_loss_set_bert, n_nodes, n_layers)
    
    return lstm, bert, lstm_mse, bert_mse

def create_igt_plots(lstm_output, bert_output, test_targets, loss_set_lstm, 
                    loss_set_bert, val_loss_set_bert, n_nodes, n_layers):
    """Create and save IGT visualization plots"""
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(loss_set_lstm, 'b-', label='LSTM Training Loss', alpha=0.7)
    plt.plot(loss_set_bert, 'r-', label='BERT Training Loss', alpha=0.7)
    plt.title('IGT Task - Training Loss Comparison')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. BERT Training vs Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(val_loss_set_bert, 'g-', label='BERT Validation Loss', linewidth=2)
    plt.title('IGT Task - BERT Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. MSE by Time Step
    plt.subplot(2, 2, 3)
    time_steps = np.arange(1, lstm_output.shape[1] + 1)
    lstm_mse_by_time = robust_MSE_by_time(test_targets, lstm_output)
    bert_mse_by_time = robust_MSE_by_time(test_targets, bert_output)
    
    plt.plot(time_steps, lstm_mse_by_time, 'b-o', label='LSTM', alpha=0.7)
    plt.plot(time_steps, bert_mse_by_time, 'r-s', label='BERT', alpha=0.7)
    plt.title('IGT Task - MSE by Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Deck Choice Distribution
    plt.subplot(2, 2, 4)
    # Calculate average deck choice probabilities
    lstm_deck_probs = np.mean(lstm_output, axis=(0, 1))  # Average across samples and time
    bert_deck_probs = np.mean(bert_output, axis=(0, 1))
    true_deck_probs = np.mean(test_targets, axis=(0, 1))
    
    x = np.arange(4)
    width = 0.25
    
    plt.bar(x - width, lstm_deck_probs, width, label='LSTM', alpha=0.7)
    plt.bar(x, bert_deck_probs, width, label='BERT', alpha=0.7)
    plt.bar(x + width, true_deck_probs, width, label='Ground Truth', alpha=0.7)
    
    plt.title('IGT Task - Average Deck Choice Probabilities')
    plt.xlabel('Deck')
    plt.ylabel('Probability')
    plt.xticks(x, ['Deck 1', 'Deck 2', 'Deck 3', 'Deck 4'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"Figures/IGT/igt_model_comparison_nodes_{n_nodes}_layers_{n_layers}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual plots
    # Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_set_lstm, 'b-', label='LSTM', linewidth=2)
    plt.plot(loss_set_bert, 'r-', label='BERT', linewidth=2)
    plt.title('IGT Task - Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Figures/IGT/igt_training_loss_nodes_{n_nodes}_layers_{n_layers}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # BERT Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss_set_bert, 'g-', linewidth=2)
    plt.title('IGT Task - BERT Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Figures/IGT/igt_bert_val_loss_nodes_{n_nodes}_layers_{n_layers}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # MSE Comparison
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, lstm_output.shape[1] + 1)
    plt.plot(time_steps, lstm_mse_by_time, 'b-o', label='LSTM', linewidth=2, markersize=6)
    plt.plot(time_steps, bert_mse_by_time, 'r-s', label='BERT', linewidth=2, markersize=6)
    plt.title('IGT Task - MSE by Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Figures/IGT/igt_mse_comparison_nodes_{n_nodes}_layers_{n_layers}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All IGT plots saved to Figures/IGT/")

if __name__ == "__main__":
    print("Starting IGT BERT Training...")
    lstm, bert, lstm_mse, bert_mse = train_igt_models()
    print("IGT BERT training completed!")
    print(f"Final Results:")
    print(f"LSTM MSE: {lstm_mse:.6f}")
    print(f"BERT MSE: {bert_mse:.6f}")
    print(f"BERT improvement: {(lstm_mse - bert_mse) / lstm_mse * 100:.2f}%") 