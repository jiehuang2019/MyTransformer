import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=100):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]

        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
        x = self.transformer(x, mask)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]

        return self.lm_head(x)  # [batch, seq_len, vocab_size]

    def generate(self, input_ids, max_new_tokens=10):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)  # [batch, seq_len, vocab]
            next_token_logits = logits[:, -1, :]  # get last token logits
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

vocab_size = 100
model = MiniGPT(vocab_size)
input_ids = torch.randint(0, vocab_size, (1, 5))  # [batch, seq_len]
output = model.generate(input_ids, max_new_tokens=5)
print(input_ids)
print(output)

