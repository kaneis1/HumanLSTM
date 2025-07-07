## Python Imports

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import seaborn as sns
import math

sns.set()
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

MSE = mean_squared_error
# For v3: we use 8 moves to predict the 9th move
input_length = 8
prediction_length = 1

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

## IPD Task v3 - Predict last move using previous 8 moves

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

def ipd_regression_data_v3(regressiondata):
    """Create regression data for v3: use 8 moves to predict the 9th move"""
    x, y = [], []
    for e in regressiondata:
        # Features: previous 8 moves from both players + additional features
        features = []
        
        # Previous 8 moves from both players (positions 0-7)
        for t in range(8):
            features.extend([e[14 - t], e[23 - t]])  # Player 0 and Player 1 moves
        
        # Additional features from the original regression
        features.extend([
            e[2], e[3], e[0], e[4], e[1],  # Basic features
            e[1] * e[2], e[1] * e[3],       # Interaction features
            e[46], e[47],                    # Additional features
            e[46] * e[1]                     # More interaction features
        ])
        
        x.append(features)
        y.append(np.abs(e[13]))  # Target: the 9th move (position 8)
    
    return np.array(x), np.array(y)

def valid_ipd_v3(n):
    """Load and prepare data for v3 prediction task"""
    if n < 1:
        n = int(8258 * n)
    shuffindex = np.random.permutation(8258)
    data = pd.read_csv("data/IPD/all_data.csv")
    trajs = np.array(data[data["period"] == 10].iloc[:, 9:27])  # (8258, 18)
    regressiondata = np.array(data[data["period"] == 10].iloc[:, 3:51])  # (8258, 48)
    regressiondata, trajs = regressiondata[shuffindex], trajs[shuffindex]
    
    # Create regression data for v3
    train_set_rgx, train_set_rgy = ipd_regression_data_v3(regressiondata[n:])
    test_set_rgx, test_set_rgy = ipd_regression_data_v3(regressiondata[:n])
    
    # Prepare trajectory data for neural networks
    trajs = trajs.reshape((trajs.shape[0], 2, 9))  # (8258, 2, 9)
    trajs[trajs == 0] = 2
    trajs = trajs - 1
    train_set, test_set = trajs[n:], trajs[:n]
    
    return train_set, test_set, train_set_rgx, test_set_rgx, train_set_rgy, test_set_rgy

def getCR(r):  # cooperation rate
    if len(r.shape) == 3:
        r = r[:, :, 0]
    cr = np.zeros(r.shape)
    for t in np.arange(r.shape[1]):
        for b in np.arange(r.shape[0]):
            cr[b, t] = r[b, : t + 1].sum() / (t + 1)
    return cr

def ipd_set2arset_v3():
    """Prepare AR data for v3: use 8 moves to predict the 9th move"""
    null_arset = np.zeros((9, 2))
    train_arset = np.zeros((9, 2))
    for ins in train_set:
        ts = ins.copy()
        ts[ts == 0] = -1
        train_arset = np.concatenate((train_arset, null_arset, ts.T), axis=0)
    return train_arset

class lstmModel(nn.Module):
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
        out = nn.Softmax(dim=-1)(out)
        return out

class bertModel(nn.Module):
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
        x = nn.Softmax(dim=-1)(x)
        
        return x

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

n_fold = 5
for fold in np.arange(n_fold):
    (
        train_set,
        test_set,
        train_set_rgx,
        test_set_rgx,
        train_set_rgy,
        test_set_rgy,
    ) = valid_ipd_v3(0.4)
    full_data = {
        "train": train_set,
        "test": test_set,
        "train_set_rgx": train_set_rgx,
        "train_set_rgy": train_set_rgy,
        "test_set_rgx": test_set_rgx,
        "test_set_rgy": test_set_rgy,
    }
    with open("data/IPD/processed_train_test_v3.pkl", "wb") as handle:
        pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    n_nodes, n_layers = 10, 2
    
    # Train LSTM model
    lstm = lstmModel(2, n_nodes, 2, n_layers).to(device)
    criterion_lstm = nn.MSELoss()
    optimizer_lstm = optim.Adam(lstm.parameters(), lr=1e-2)
    n_epochs, window, batch_size = 10, 10, 100
    loss_set_lstm = []
    
    print("Training LSTM model...")
    for ep in np.arange(n_epochs):
        for bc in np.arange(train_set.shape[0] / batch_size):
            # Use first 8 moves as input, predict the 9th move
            inputs = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
                .to(device)
            )
            # Input: first 8 moves (positions 0-7)
            input_data = inputs[:, :input_length, :]  # (batch_size, 8, 2)
            
            # Target: 9th move (position 8)
            target = inputs[:, input_length, 0].unsqueeze(1)  # (batch_size, 1)
            
            output = lstm(input_data)
            # Take the last output and predict the 9th move
            pred = output[:, -1, 0]  # (batch_size, 2) - probabilities for both actions
         
            
            loss = criterion_lstm(pred, target.squeeze())
            optimizer_lstm.zero_grad()
            loss.backward()
            optimizer_lstm.step()
            print_loss = loss.item()
            loss_set_lstm.append(print_loss)
            if bc % window == 0:
                print(fold)
                print(
                    "LSTM Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set.shape[0] / batch_size,
                        print_loss,
                    )
                )
    lstm = lstm.eval()
    
    # Train BERT model
    bert = bertModel(2, n_nodes, 2, n_layers).to(device)
    criterion_bert = nn.MSELoss()
    optimizer_bert = optim.Adam(bert.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_set_bert = []
    val_loss_set_bert = []
    
    print("Training BERT model...")
    for ep in np.arange(n_epochs):
        bert.train()
        epoch_losses = []
        
        for bc in np.arange(train_set.shape[0] / batch_size):
            # Use first 8 moves as input, predict the 9th move
            inputs = Variable(
                torch.from_numpy(
                    train_set[int(bc * batch_size) : int((bc + 1) * batch_size)]
                )
                .transpose(1, 2)
                .float()
                .to(device)
            )
            # Input: first 8 moves (positions 0-7)
            input_data = inputs[:, :input_length, :]  # (batch_size, 8, 2)
            
            # Target: 9th move (position 8)
            target = inputs[:, input_length, 0]  # (batch_size,) - the 9th move
            
            output = bert(input_data) 
            # Take the last output and predict the 9th move
            pred = output[:, -1, 0]  # (batch_size,) - probability of cooperation for 9th move
            
            loss = criterion_bert(pred, target)
            optimizer_bert.zero_grad()
            loss.backward()
            optimizer_bert.step()
            print_loss = loss.item()
            epoch_losses.append(print_loss)
            loss_set_bert.append(print_loss)
            
            if bc % window == 0:
                print(fold)
                print(
                    "BERT Epoch[{}/{}], Batch[{}/{}], Loss: {:.5f}".format(
                        ep + 1,
                        n_epochs,
                        bc + 1,
                        train_set.shape[0] / batch_size,
                        print_loss,
                    )
                )
        
        # Validation step
        bert.eval()
        with torch.no_grad():
            val_inputs = Variable(torch.from_numpy(test_set).transpose(1, 2).float().to(device))
            val_input_data = val_inputs[:, :input_length, :]  # First 8 moves
            val_output = bert(val_input_data)
            val_pred = val_output[:, -1, 0]  # Probability of cooperation for 9th move
            val_targ = val_inputs[:, input_length, 0]  # Actual 9th move
            val_loss = criterion_bert(val_pred, val_targ).item()
            val_loss_set_bert.append(val_loss)
            
            if ep % 2 == 0:  # Print every 2 epochs
                print(f"BERT Epoch {ep+1}: Train Loss: {np.mean(epoch_losses):.5f}, Val Loss: {val_loss:.5f}")
    
    bert = bert.eval()
    
    # Save the trained BERT model
    torch.save(bert.state_dict(), f"bert_model_v3_fold_{fold}.pth")
    print(f"BERT model saved for fold {fold}")
    
    # AR model
    train_arset = ipd_set2arset_v3()
    armodel = VAR(train_arset)
    armodel = armodel.fit()
    px = torch.from_numpy(test_set).transpose(1, 2).float().to(device)
    ry = px
    
    # Predict the 9th move using AR
    pyar = np.zeros((px.shape[0], 1))  # Only predict the 9th move
    for i in np.arange(px.shape[0]):
        # Use first 8 moves to predict the 9th move
        forecast = armodel.forecast(np.array(px[i, :input_length].cpu()), 1)
        pyar[i, 0] = 1 if forecast[0][0] > 0 else 0
    
    # LR model
    lrmodel = LogisticRegression(random_state=0, max_iter=1000).fit(
        train_set_rgx, train_set_rgy
    )
    pylrraw = lrmodel.predict(test_set_rgx)
    
    # Calculate the correct shape based on actual test set size
    num_test_games = test_set.shape[0]
    pylr = pylrraw.reshape((num_test_games, 1))  # Only predict the 9th move
    
    # Get predictions from both neural models
    varX = Variable(px)
    # For LSTM: use first 8 moves to predict 9th move
    input_lstm = varX[:, :input_length, :]
    py_lstm_raw = lstm(input_lstm).squeeze().data.cpu().numpy()
    py_lstm = py_lstm_raw[:, -1, 0]  # Probability of cooperation for 9th move
    
    # For BERT: use first 8 moves to predict 9th move
    input_bert = varX[:, :input_length, :]
    py_bert_raw = bert(input_bert).squeeze().data.cpu().numpy()
    py_bert = py_bert_raw[:, -1, 0]  # Probability of cooperation for 9th move
    
    if fold == 0:
        test_set_full = test_set
        py_lstm_full = py_lstm
        py_bert_full = py_bert
        pyar_full = pyar
        pylr_full = pylr
    else:
        test_set_full = np.concatenate((test_set_full, test_set))
        px = torch.from_numpy(test_set_full).transpose(1, 2).float()
        ry = px
        py_lstm_full = np.concatenate((py_lstm_full, py_lstm))
        py_bert_full = np.concatenate((py_bert_full, py_bert))
        pyar_full = np.concatenate((pyar_full, pyar))
        pylr_full = np.concatenate((pylr_full, pylr))
        py_lstm = py_lstm_full
        py_bert = py_bert_full
        pyar = pyar_full
        pylr = pylr_full

# Calculate cooperation rates for the 9th move prediction
ryc = ry[:, input_length, 0]  # Actual 9th move
pyc = py_lstm  # LSTM prediction for 9th move
pycar = pyar[:, 0]  # AR prediction for 9th move
pyclr = pylr[:, 0]  # LR prediction for 9th move
pyc_bert = py_bert  # BERT prediction for 9th move

# Convert tensor to numpy array for calculations
ryc_np = ryc.cpu().numpy() if torch.is_tensor(ryc) else ryc

# Calculate MSE for single prediction
bert_mse = MSE(ryc_np, pyc_bert)
lstm_mse = MSE(ryc_np, py_lstm)
ar_mse = MSE(ryc_np, pycar)
lr_mse = MSE(ryc_np, pyclr)

print(f"BERT MSE: {bert_mse:.6f}")
print(f"LSTM MSE: {lstm_mse:.6f}")
print(f"AR MSE: {ar_mse:.6f}")
print(f"LR MSE: {lr_mse:.6f}")

# Create figures directory if it doesn't exist
import os
os.makedirs("Figures/IPD_v3", exist_ok=True)

# Plot cooperation rates comparison
plt.clf()
plt.figure(figsize=(10, 6))
models = ['LSTM', 'BERT', 'AR', 'LR', 'Human']
coop_rates = [np.mean(pyc), np.mean(pyc_bert), np.mean(pycar), np.mean(pyclr), np.mean(ryc_np)]
coop_stds = [np.std(pyc), np.std(pyc_bert), np.std(pycar), np.std(pyclr), np.std(ryc_np)]

x_pos = np.arange(len(models))
plt.bar(x_pos, coop_rates, yerr=coop_stds, capsize=5, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Cooperation Rate')
plt.title('IPD v3 - 9th Move Cooperation Prediction')
plt.xticks(x_pos, models)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(f"Figures/IPD_v3/ipd_v3_coop_comparison_nodes_{n_nodes}_layers_{n_layers}.png")

# Plot MSE comparison
plt.clf()
plt.figure(figsize=(10, 6))
mse_values = [lstm_mse, bert_mse, ar_mse, lr_mse]
mse_models = ['LSTM', 'BERT', 'AR', 'LR']

x_pos = np.arange(len(mse_models))
plt.bar(x_pos, mse_values, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('IPD v3 - 9th Move Prediction MSE')
plt.xticks(x_pos, mse_models)
plt.tight_layout()
plt.savefig(f"Figures/IPD_v3/ipd_v3_mse_comparison_nodes_{n_nodes}_layers_{n_layers}.png")

# Plot training losses
plt.clf()
plt.plot(loss_set_lstm, label='LSTM Loss')
plt.plot(loss_set_bert, label='BERT Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('IPD v3 - Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f"Figures/IPD_v3/ipd_v3_training_loss_nodes_{n_nodes}_layers_{n_layers}.png")

# Plot validation loss for BERT
plt.clf()
plt.plot(val_loss_set_bert, 'r-', label='BERT Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('IPD v3 - BERT Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f"Figures/IPD_v3/ipd_v3_bert_val_loss_nodes_{n_nodes}_layers_{n_layers}.png")

print(f"BERT vs LSTM improvement: {(lstm_mse - bert_mse) / lstm_mse * 100:.2f}%")
print(f"BERT vs AR improvement: {(ar_mse - bert_mse) / ar_mse * 100:.2f}%")
print(f"BERT vs LR improvement: {(lr_mse - bert_mse) / lr_mse * 100:.2f}%") 