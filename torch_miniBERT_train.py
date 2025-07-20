import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
class MiniBERTClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d_model=256, nhead=4, num_layers=4, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = x.transpose(0, 1)

        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)

        cls_rep = self.norm(x[:, 0])  # like BERT's [CLS] token
        return self.head(cls_rep)

def train(model, tokenizer, train_texts, train_labels, epochs=3, lr=3e-4):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    # Tokenize
    enc = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(train_labels)
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        round=0
        for input_ids, attn_mask, label in loader:
            device = next(model.parameters()).device
            input_ids, attn_mask, label = input_ids.to(device), attn_mask.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}: Round {round+1}: loss={total_loss:.4f}")
            round=round+1
            if(round>10):
               break
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("sst2")  # or "imdb" for movie reviews
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#train_texts = dataset["train"]["sentence"]
#train_labels = dataset["train"]["label"]
#fix
train_texts = list(dataset["train"]["sentence"])
train_labels = list(dataset["train"]["label"])


model = MiniBERTClassifier(vocab_size=tokenizer.vocab_size)
train(model, tokenizer, train_texts, train_labels)
# Save model weights only
torch.save(model.state_dict(), "mini_bert_classifier.pt")
# Make sure model class definition is available
model = MiniBERTClassifier(
    vocab_size=tokenizer.vocab_size,  # same as training
    num_classes=2                     # or whatever you used
)
model.load_state_dict(torch.load("mini_bert_classifier.pt"))
model.eval()

def predict(texts):
    # Tokenize input
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Send to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    return preds, probs

texts = ["I love this movie!", "Worst experience ever."]
preds, probs = predict(texts)

for text, p, prob in zip(texts, preds, probs):
    print(f"{text} → class {p.item()} | confidence {prob.max().item():.4f}")

