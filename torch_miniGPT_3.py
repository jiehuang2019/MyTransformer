import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = x.transpose(0, 1)

        # Causal mask for left-to-right generation
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)

        # src_key_padding_mask: [batch, seq_len]
        padding_mask = (attention_mask == 0) if attention_mask is not None else None

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=10):
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids, attention_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                new_mask = torch.ones_like(next_token)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        return input_ids

prompt = ["Once upon a time"]
batch = tokenizer(prompt, return_tensors="pt", padding=True)
model = MiniGPT(tokenizer.vocab_size)
generated = model.generate(batch["input_ids"], batch["attention_mask"], max_new_tokens=10)
print(prompt)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))

