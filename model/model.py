import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import PositionalEncoding
from model.transformer import SelfAttention, CrossAttention
from model.sincnet import SincConv1D

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # First hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SincNetRNN(nn.Module):
    
    def __init__(self, sinc_out_channels=40, sinc_kernel_size=251, sample_rate=8000,
                 hidden_size=64, num_layers=3, num_classes=4):
        super(SincNetRNN, self).__init__()

        # SincNet front-end replaces MFCC
        self.sinc = SincConv1D(out_channels=sinc_out_channels,
                               kernel_size=sinc_kernel_size,
                               sample_rate=sample_rate)
        self.bn_sinc = nn.BatchNorm1d(sinc_out_channels)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.bn_pooled = nn.BatchNorm1d(sinc_out_channels)
        
        self.pe_rnn = PositionalEncoding(d_model=hidden_size)
        self.pe_sinc = PositionalEncoding(d_model=sinc_out_channels)
        
        self.rnn = nn.LSTM(input_size=sinc_out_channels,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=0.3,
                           batch_first=True)
        
        self.ln_rnn = nn.LayerNorm(hidden_size)
        
        self.self_attn_rnn = SelfAttention(dim=hidden_size, heads=4)
        self.self_attn_sinc = SelfAttention(dim=sinc_out_channels, heads=4)
        
        self.cross_attn_sinc_to_rnn = CrossAttention(
            query_dim=sinc_out_channels,
            context_dim=hidden_size,
            num_heads=4,
            head_dim=32
        )
        
        self.cross_attn_rnn_to_sinc = CrossAttention(
            query_dim=hidden_size,
            context_dim=sinc_out_channels,
            num_heads=4,
            head_dim=32
        )
        
        self.classifier = MLPClassifier(input_dim=256, num_classes=4)
        
        self.dropout_cls = nn.Dropout(0.3)
        
    def forward(self, waveform):
        
        waveform = waveform.unsqueeze(1).to(torch.float32)

        # normalizing the input waveform
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        waveform = waveform / (waveform.std(dim=-1, keepdim=True) + 1e-9)
        
        # seismic waveform: shape (B, 1, waveform_len)
        sinc_out = self.sinc(waveform)  # -> (B, C, T)
        
        # print("*"*10)
        # print("Shape from sincnet :", sinc_out.shape)
        # print("*"*10)
        
        sinc_out = self.bn_sinc(sinc_out)                        # BatchNorm on channels
        sinc_out = F.relu(sinc_out)                              # Activation
        sinc_out = self.pool(sinc_out)                           # → (B, C, 16)
        
        # print("Shape from pool of sincnet feat :", sinc_out.shape)
        
        sinc_out = self.bn_pooled(sinc_out)
        sinc_out = F.relu(sinc_out)
        
        sinc_out = sinc_out.transpose(1, 2)  # -> (B, T, C)
        
        
        # print("The shape from pooling which will send to RNN :", sinc_out.shape)
        # Feed to RNN
        rnn_out, _ = self.rnn(sinc_out)
        rnn_out = self.ln_rnn(rnn_out)     # normalizing RNN op
        
        # print(rnn_out.shape)
        
        # positional encoding RNN output
        rnn_out_pe = self.pe_rnn(rnn_out)
        # positional encoding sinc_conv output
        sinc_out_pe = self.pe_sinc(sinc_out)

        # Self attention on both branches
        sinc_out_sa = self.self_attn_sinc(sinc_out_pe)
        rnn_out_sa = self.self_attn_rnn(rnn_out_pe)
        
        # Cross attention
        attn1 = self.cross_attn_sinc_to_rnn(sinc_out_sa, rnn_out_sa)  # (B, T, D)
        attn2 = self.cross_attn_rnn_to_sinc(rnn_out_sa, sinc_out_sa)  # (B, T, D)
        
        # Combine (e.g., mean or max pool)
        attn1_pooled = attn1.mean(dim=1)  # [B, D]
        attn2_pooled = attn2.mean(dim=1)
    
        # last_hidden = fused.mean(dim=1)  # mean pooling
        last_hidden = torch.cat([attn1_pooled, attn2_pooled], dim=-1)  # [B, 2D]
        last_hidden = self.dropout_cls(last_hidden)
        out_logits = self.classifier(last_hidden)

        return out_logits