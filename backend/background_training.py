import asyncio
import json
import os
from datetime import datetime

import torch

from ai_model import MiniCompanionAI
from data_mixer import DataMixer
from memory_grid import MemoryGrid
from tokenizer import tokenize, normalize_lang

memory = MemoryGrid()
mixer = DataMixer()

# Global tokenizer vocab and reverse (shared with main.py if imported there)
tokenizer_vocab = None
reverse_vocab = None


def get_unique_words_count() -> int:
    """Count unique words stored in the memory grid."""
    unique = set()
    for row in memory.grid:
        for col in row:
            for token_info in col:
                # token_info has 'stem' and 'original'
                unique.add(token_info['stem'])
                unique.add(token_info['original'])
    return len(unique)


def train_model():
    """
    Train the transformer on all texts from the data mixer.
    Saves model and tokenizer vocab.
    """
    global tokenizer_vocab, reverse_vocab
    texts = mixer.texts
    if not texts:
        print("No training data found.")
        return

    # Build vocabulary from all texts
    tokenizer_vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    reverse_vocab = {0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>"}

    for text in texts:
        tokens = tokenize(text, "en")
        for t in tokens:
            for w in [t['stem'], t['original']]:
                if w not in tokenizer_vocab:
                    tokenizer_vocab[w] = len(tokenizer_vocab)
                    reverse_vocab[tokenizer_vocab[w]] = w

    # Encode all documents
    encoded_docs = []
    for text in texts:
        tokens = tokenize(text, "en")
        ids = [tokenizer_vocab["<start>"]]
        for t in tokens:
            word = t['stem'] if t['stem'] in tokenizer_vocab else (
                t['original'] if t['original'] in tokenizer_vocab else "<unk>"
            )
            ids.append(tokenizer_vocab.get(word, 1))
        ids.append(tokenizer_vocab["<end>"])
        encoded_docs.append(ids)

    if not encoded_docs:
        print("No encoded docs.")
        return

    # Initialize model
    model = MiniCompanionAI(len(tokenizer_vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Training loop (simplified, can adjust epochs)
    epochs = 30
    batch_size = 4
    print(f"Starting training on {len(encoded_docs)} docs, vocab size {len(tokenizer_vocab)}...")

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(encoded_docs), batch_size):
            batch = encoded_docs[i:i + batch_size]
            if not batch:
                continue

            # Pad batch to max length
            max_len = max(len(seq) for seq in batch)
            x = torch.zeros((len(batch), max_len - 1), dtype=torch.long)
            y = torch.zeros((len(batch), max_len - 1), dtype=torch.long)

            for j, seq in enumerate(batch):
                x[j, :len(seq) - 1] = torch.tensor(seq[:-1])
                y[j, :len(seq) - 1] = torch.tensor(seq[1:])

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, len(tokenizer_vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(encoded_docs), 1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model and vocab
    torch.save(model.state_dict(), 'companion_model.pth')
    with open('tokenizer_vocab.json', 'w') as f:
        json.dump({"vocab": tokenizer_vocab, "reverse": reverse_vocab}, f)

    print("Training complete. Model saved.")


async def auto_train_monitor():
    """Background task that watches for new words in memory grid and retrains."""
    last_count = get_unique_words_count()
    print(f"Auto-train monitor started. Unique words: {last_count}")

    while True:
        await asyncio.sleep(60)  # check every minute
        current_count = get_unique_words_count()
        if current_count > last_count:
            print(f"New words detected ({current_count} vs {last_count}). Starting training...")
            await asyncio.to_thread(train_model)
            last_count = current_count
        else:
            last_count = current_count  # in case of deletion, keep sync


def start_background_training():
    """Run the monitor in the current event loop."""
    loop = asyncio.get_event_loop()
    loop.create_task(auto_train_monitor())