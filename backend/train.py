import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim

from ai_model import MiniCompanionAI
from tokenizer import tokenize, normalize_lang

# -------------------------------
# 1. Load text data from all .txt files in data_dir
# -------------------------------
def load_texts(data_dir="data"):
    texts = []
    if not os.path.isdir(data_dir):
        return texts
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                texts.extend([line.strip() for line in lines if line.strip()])
    return texts


# -------------------------------
# 2. Build vocabulary from tokenizer stems & originals
# -------------------------------
def build_vocab(texts, lang="en"):
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    reverse = {0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>"}

    for text in texts:
        tokens = tokenize(text, lang)
        for tok in tokens:
            stem = tok["stem"]
            if stem not in vocab:
                vocab[stem] = len(vocab)
                reverse[vocab[stem]] = stem
            orig = tok["original"]
            if orig != stem and orig not in vocab:
                vocab[orig] = len(vocab)
                reverse[vocab[orig]] = orig
    return vocab, reverse


# -------------------------------
# 3. Encode all texts into token ID lists
# -------------------------------
def encode_texts(texts, vocab, lang="en"):
    encoded = []
    for text in texts:
        tokens = tokenize(text, lang)
        ids = [vocab["<start>"]]
        for tok in tokens:
            word = tok["stem"] if tok["stem"] in vocab else (
                tok["original"] if tok["original"] in vocab else "<unk>"
            )
            ids.append(vocab[word])
        ids.append(vocab["<end>"])
        encoded.append(ids)
    return encoded


# -------------------------------
# 4. Batch generator
# -------------------------------
def get_batch(encoded_docs, batch_size, seq_len, vocab):
    while True:
        inputs, targets = [], []
        for _ in range(batch_size):
            doc = random.choice(encoded_docs)
            if len(doc) < seq_len + 1:
                doc = doc + [vocab["<pad>"]] * (seq_len + 1 - len(doc))
            start = random.randint(0, len(doc) - seq_len - 1)
            chunk = doc[start : start + seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
        yield torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# -------------------------------
# 5. Main training function
# -------------------------------
def train(
    data_dir="data",
    epochs=100,
    batch_size=8,
    seq_len=32,
    lr=1e-4,
    save_model="companion_model.pth",
    save_vocab="tokenizer_vocab.json",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = load_texts(data_dir)
    if not texts:
        raise RuntimeError(f"No training data found in {data_dir}")

    print(f"Loaded {len(texts)} lines of training data.")

    vocab, reverse = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab)}")

    encoded_docs = encode_texts(texts, vocab)
    print(f"Encoded {len(encoded_docs)} documents.")

    model = MiniCompanionAI(vocab_size=len(vocab))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = max(1, min(50, len(encoded_docs) // batch_size))  # adaptive steps
        batch_iterator = get_batch(encoded_docs, batch_size, seq_len, vocab)
        for step in range(steps):
            x, y = next(batch_iterator)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_model)
    with open(save_vocab, "w") as f:
        json.dump({"vocab": vocab, "reverse": reverse}, f)
    print(f"Model saved to {save_model}, vocab saved to {save_vocab}")


if __name__ == "__main__":
    train()