import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BERTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.transformer_encoder(x)
        logits = self.lm_head(x)  # Predict masked tokens
        return logits

vocab_size = 30522
model = BERTMini(vocab_size)

input_ids = torch.randint(0, vocab_size, (2, 32))  # batch of 2, 32 tokens
logits = model(input_ids)
print(logits.shape)  # [2, 32, vocab_size]
print(logits)  # [2, 32, vocab_size]

