
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer
class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = x.transpose(0, 1)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=10):
        for _ in range(max_new_tokens):
            attn_mask = torch.ones_like(input_ids)
            logits = self.forward(input_ids, attn_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have [PAD]

block_size = 128

def tokenize_and_chunk(example):
    print(example)
    if not example:
      print('+++++++')
      return None
    tokens = tokenizer(example["text"], return_attention_mask=False)["input_ids"]
    print(len(tokens))
    if len(tokens)==0:
      print('-------')
    block_size = 128
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    chunks = []
    for i in range(0, len(tokens), block_size):
        print(len(chunks))
        chunk = tokens[i : i + block_size][:]
        if len(chunk) < block_size:
            # Only pad the last chunk
            pad=[pad_id] * (block_size - len(chunk))
            print(len(chunk))
            print(len(pad))
            chunk.extend(pad)
            print(len(chunk))
        if len(chunk)!=block_size:
            print("error------")
        chunks.extend([chunk])
        print(len(chunks))
    print("before return")
    print(len(chunks))
    #return {"input_ids": chunks}
    if len(chunks)>0:
      return {"input_ids": chunks[0]}
    else:
      return {"input_ids": []}
    #return chunks
print(dataset.shape)
print(dataset[1])
print(dataset[2])
tokenized = dataset.map(tokenize_and_chunk, batched=False, remove_columns=["text"])
print(len(tokenized))
print((tokenized[1]))
print((tokenized[2]))
tokenized = tokenized.flatten_indices()  # ensures each example is one chunk
print(len(tokenized))
print((tokenized[1]))
print((tokenized[2]))
#tokenized.set_format(type="torch")
tokenized.set_format(type="torch", columns=["input_ids"])
print(len(tokenized))
print((tokenized[1]))
print((tokenized[2]))

from torch.utils.data import DataLoader
from torch.optim import AdamW

model = GPTMini(vocab_size=tokenizer.vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from torch.utils.data import DataLoader

'''
def collate_fn(batch):
    # batch is a list of dicts, each with "input_ids"
    #[print((x["input_ids"])) for x in batch]
    #input_ids = [torch.cat(x["input_ids"]) for x in batch]
    input_ids = [torch.cat(x) for x in batch]
    return {
        "input_ids": torch.stack(input_ids)
    }

train_loader = DataLoader(
    tokenized,               # ✅ full dataset, not tokenized["input_ids"]
    batch_size=16,
    shuffle=True,
    #collate_fn=collate_fn
)

'''
train_loader = DataLoader(
    tokenized,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda batch: torch.stack([x["input_ids"] for x in batch if len(x["input_ids"])> 0 ])
    #collate_fn=lambda batch: torch.stack([x["input_ids"] for x in batch])
    #collate_fn=lambda batch: torch.stack([torch.tensor(x) for x in batch])
)



optimizer = AdamW(model.parameters(), lr=5e-4)

for epoch in range(30):
    model.train()
    total_loss = 0
    round=0
    for batch in train_loader:
        print(type(batch))
        input_ids = batch.to(device)

        labels = input_ids.clone()

        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        logits = model(input_ids, attention_mask)

        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, tokenizer.vocab_size),
                                           labels[:, 1:].reshape(-1),
                                           ignore_index=tokenizer.pad_token_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1}: Round {round+1} - Loss: {total_loss / (round+1):.4f}")
        round=round+1
        if round>1000:
           break
    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

prompt = "In the beginning"
tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens["input_ids"].to(device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0], skip_special_tokens=True))

