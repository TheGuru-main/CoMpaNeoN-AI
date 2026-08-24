import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import MiniCompanionAI
from tokenizer import tokenize, normalize_lang

# -------------------------------
# 1. Load and prepare training data
# -------------------------------
def load_texts(data_dir="data"):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                texts.extend([line.strip() for line in lines if line.strip()])
    return texts

# -------------------------------
# 2. Build tokenizer vocabulary
# -------------------------------
def build_vocab(texts, lang="en"):
    """Return dict mapping token (stem or original) -> int id, and reverse dict."""
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    reverse = {0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>"}

    for text in texts:
        tokens = tokenize(text, lang)
        for tok in tokens:
            # Use stem as the primary token for training
            word = tok["stem"]
            if word not in vocab:
                vocab[word] = len(vocab)
                reverse[vocab[word]] = word
            # Also add original if different
            orig = tok["original"]
            if orig != word and orig not in vocab:
                vocab[orig] = len(vocab)
                reverse[vocab[orig]] = orig
    return vocab, reverse

# -------------------------------
# 3. Encode texts into ID lists
# -------------------------------
def encode_texts(texts, vocab, lang="en"):
    encoded = []
    for text in texts:
        tokens = tokenize(text, lang)
        ids = []
        # Add start token
        ids.append(vocab["<start>"])
        for tok in tokens:
            # Prefer stem; if not present, original; else unk
            word = tok["stem"] if tok["stem"] in vocab else (tok["original"] if tok["original"] in vocab else "<unk>")
            ids.append(vocab[word])
        # Add end token
        ids.append(vocab["<end>"])
        encoded.append(ids)
    return encoded

# -------------------------------
# 4. Batch generator
# -------------------------------
def get_batch(encoded_docs, batch_size, seq_len, vocab):
    """Generate random batches of input-target pairs."""
    while True:
        inputs, targets = [], []
        for _ in range(batch_size):
            doc = random.choice(encoded_docs)
            if len(doc) < seq_len + 1:
                # Pad if too short
                doc = doc + [vocab["<pad>"]] * (seq_len + 1 - len(doc))
            start_idx = random.randint(0, len(doc) - seq_len - 1)
            chunk = doc[start_idx : start_idx + seq_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
        yield torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# -------------------------------
# 5. Main training loop
# -------------------------------
def train(data_dir="data", epochs=100, batch_size=8, seq_len=32, lr=1e-4, save_model="companion_model.pth", save_vocab="tokenizer_vocab.json"):
    texts = load_texts(data_dir)
    if not texts:
        raise RuntimeError("No training data found in " + data_dir)

    print(f"Loaded {len(texts)} lines of training data.")

    vocab, reverse = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab)}")

    encoded_docs = encode_texts(texts, vocab)
    print(f"Encoded {len(encoded_docs)} documents.")

    model = MiniCompanionAI(vocab_size=len(vocab))
    model.to(device if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 50  # arbitrary number of steps per epoch
        batch_iterator = get_batch(encoded_docs, batch_size, seq_len, vocab)
        for step in range(steps):
            x, y = next(batch_iterator)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model and vocab
    torch.save(model.state_dict(), save_model)
    with open(save_vocab, 'w') as f:
        json.dump({"vocab": vocab, "reverse": reverse}, f)
    print(f"Model saved to {save_model}, vocab saved to {save_vocab}")

if __name__ == "__main__":
    train()