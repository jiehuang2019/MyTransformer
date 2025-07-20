import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=2, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask: [batch, seq_len]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # [batch, seq_len] → [seq_len, batch, d_model]
        x = x.transpose(0, 1)

        # create src_key_padding_mask: [batch, seq_len]
        padding_mask = (attention_mask == 0)  # 0 = pad → True

        output = self.encoder(x, src_key_padding_mask=padding_mask)
        cls_token = output[0]  # use the first token
        return self.cls_head(cls_token)
inputs=["I love it", "Terrible movie!!!"]
batch = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
model = MiniBERT(tokenizer.vocab_size)
logits = model(batch["input_ids"], batch["attention_mask"])
print(inputs)
print(torch.argmax(logits, dim=-1))  # tensor([1, 0]) maybe

