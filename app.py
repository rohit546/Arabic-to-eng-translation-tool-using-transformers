import streamlit as st
import torch
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.wq(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        
        # Concatenate heads and pass through final linear layer
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(out)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Encoder-Decoder attention
        attn_output = self.enc_dec_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # Define d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)  # Now self.d_model is defined
        src = self.pos_encoding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# Decoder
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # Define d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # Now self.d_model is defined
        tgt = self.pos_encoding(tgt)
        for layer in self.layers:
            tgt = layer(tgt, enc_output, src_mask, tgt_mask)
        return tgt

# Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)

# Load vocabularies
with open("src_vocab.json", "r") as f:
    src_vocab = json.load(f)
with open("tgt_vocab.json", "r") as f:
    tgt_vocab = json.load(f)
tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}  # Inverse vocabulary for decoding

# Load the model
device = torch.device("cpu")  # Use CPU for inference
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load("transformer_nmt.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Preprocess input text
def preprocess_input(text, vocab):
    tokens = text.split() + ["<eos>"]
    indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    return torch.LongTensor(indices).unsqueeze(0).to(device)  # Add batch dimension

# Generate translation
def translate(text, max_len=100):
    # Preprocess input
    src = preprocess_input(text, src_vocab)
    
    # Initialize target sequence with <sos>
    tgt = torch.LongTensor([[tgt_vocab["<sos>"]]]).to(device)
    
    # Generate translation token by token
    for _ in range(max_len):
        with torch.no_grad():  # Disable gradient calculation
            output = model(src, tgt)
        next_token = output.argmax(-1)[:, -1:]  # Greedy decoding
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if <eos> is generated
        if next_token.item() == tgt_vocab["<eos>"]:
            break
    
    # Convert indices to tokens
    translation = " ".join([tgt_vocab_inv.get(idx.item(), "") for idx in tgt[0][1:-1]])
    return translation

# Streamlit app
st.title("Arabic to English Translator")

# Input text box
input_text = st.text_area("Enter Arabic text:", "مرحبًا كيف حالك؟")

# Translate button
if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some Arabic text.")
    else:
        # Generate translation
        translated_text = translate(input_text)
        
        # Display translation
        st.success("Translation:")
        st.write(translated_text)
