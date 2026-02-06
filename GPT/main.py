#This creates a streamlit frontend page that allows you talk with this gpt model recreation
#To run this, type "streamlit run main.py" in the terminal

import torch
import tiktoken
import torch.nn as nn
import numpy as np 
import pandas as pd
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch
import math
import streamlit as st

tokenizer =  tiktoken.get_encoding("gpt2")
def text_to_tokens(text):
    text = tokenizer.encode(text)
    token = torch.tensor(text).unsqueeze(0)
    return token
def tokens_to_text(tokens):
    return tokenizer.decode(tokens.squeeze(0).tolist())


class Causual_attentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len,context_len), diagonal=1))

    def forward(self, x):
        b, nums_t, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_score = queries @ keys.transpose(1,2)
        causal_mask = self.mask[:nums_t, :nums_t].bool()
        attn_score = attn_score.masked_fill(causal_mask, -torch.inf)
        attn_weights = torch.softmax(attn_score/(keys.shape[-1]**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

class multiheadAttentionV1(torch.nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context)


import torch.nn as nn
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock (cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        self.out_head.weight = self.tok_emb.weight
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos)[None, :, :]
        pos_embeds = self.pos_emb(pos)
        x = tok_embeds + pos_embeds
        x= self.drop_emb(x)
        x= self.trf_blocks(x)
        x= self.final_norm(x)
        logits = self.out_head(x)
        return logits

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi , device=x.device))  * (x + 0.044715 * x ** 3)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = multiheadAttentionV1(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_len=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            bias=cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x= self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x= self.drop_shortcut(x)
        x = x + shortcut 
        

        return x

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']), GELU(), nn.Linear(4* cfg['emb_dim'],cfg['emb_dim']))

    def forward(self, x):
        return self.layers(x)
class DNN(nn.Module):
    def __init__(self, layers, shortcut):
        super().__init__()
        self.shortcut =shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layers[0],layers[1]), GELU()),
            nn.Sequential(nn.Linear(layers[1],layers[2]), GELU()),
            nn.Sequential(nn.Linear(layers[2],layers[3]), GELU()),
            nn.Sequential(nn.Linear(layers[3],layers[4]), GELU()),
            nn.Sequential(nn.Linear(layers[4],layers[5]), GELU())
        ])
    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            if self.shortcut and x.shape == output.shape:
                x = x+output
            else:
                x = output
        return x
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean)/(torch.sqrt( var+self.eps ))
        return self.scale * norm_x + self.shift
    
def generate(model, idx, max_new_tokens, context_size, temp=0.0, top_k=None, eos_id=0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temp > 0:
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return tokens_to_text(idx)

def talk(talk):
    token_ids = generate(model, text_to_tokens(talk), 30, 1024, top_k=25, temp=1.4 )
    return token_ids
GPT2_Config = {
    "vocab_size": 50257, #All Words in LLM
    'context_length': 1024, #Size of the each row in batch
    'emb_dim': 768, #Dimensions of Embeddinging Vectors
    "n_heads": 12, #Number of Transfoemrs
    "n_layers": 12, #Number of Transformer layers
    "drop_rate": 0.1, #Percentage of Tokens that are dropped
    "qkv_bias": True
}

model = GPT(GPT2_Config)             
model.load_state_dict(torch.load("gpt_state_dict.pt"))


st.title("GPT2 Model(Recreated From Scratch)")
st.markdown("Please Enter Text:")
st.divider()

data_text= st.text_input("Enter your chat:")
if st.button("talk"):
    st.success(talk(data_text))