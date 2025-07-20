import torch
import torch.nn as nn

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Transformer expects [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)

        # Causal mask: prevent tokens from attending to future tokens
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)

        x = self.transformer(x, mask)  # Self-attention with causal masking
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        logits = self.lm_head(x)
        return logits

model = GPTMini(vocab_size=50257)
input_ids = torch.randint(0, 50257, (2, 32))  # batch_size=2, seq_len=32
output = model(input_ids)  # output shape: [2, 32, vocab_size]
print(output.shape)
print(output)
