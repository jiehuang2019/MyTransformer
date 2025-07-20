import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=100, num_classes=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]

        x = self.transformer(x)  # bidirectional: no mask
        x = x[0]  # [batch, d_model], use first token (like [CLS])
        return self.cls_head(x)

vocab_size = 100
model = MiniBERT(vocab_size)
input_ids = torch.randint(0, vocab_size, (4, 10))  # [batch=4, seq_len=10]
logits = model(input_ids)
preds = torch.argmax(logits, dim=-1)
print(input_ids)
print(preds)  # class predictions

