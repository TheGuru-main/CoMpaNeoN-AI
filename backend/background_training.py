import asyncio
import json
import os
from datetime import datetime, timezone

import torch
from langdetect import detect, LangDetectException  # Added for multi-lingual rooms

from ai_model import MiniCompanionAI
from data_mixer import DataMixer
from memory_grid import MemoryGrid
from tokenizer import tokenize, normalize_lang

memory = MemoryGrid()
mixer = DataMixer()

# Global tokenizer vocab and reverse (shared with main.py if imported there)
tokenizer_vocab = None
reverse_vocab = None


def detect_lang(text: str) -> str:
    """
    Dynamically detect language for custom tokenization rules.
    Safely falls back to 'en' if text is too short or contains ambiguous characters.
    """
    try:
        if not text.strip():
            return "en"
        return detect(text)
    except LangDetectException:
        return "en"  # Safe default fallback for code snippets, emojis, or short strings


def get_unique_words_count() -> int:
    """Count unique words stored in the memory grid."""
    unique = set()
    for row in memory.grid:
        for col in row:
            for token_info in col:
                unique.add(token_info['stem'])
                unique.add(token_info['original'])
    return len(unique)


def train_model(grid_cv_params: dict = None):
    """
    Train the transformer on all texts from the data mixer with batch versioning.
    Saves unique model checkpoints and synchronized tokenizer vocab for zero-downtime hot swapping.
    """
    global tokenizer_vocab, reverse_vocab
    texts = mixer.texts
    if not texts:
        print("No training data found.")
        return

    # 1. GENERATE DEPLOYMENT VERSION STRINGS
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    batch_version = f"v_{timestamp}"
    
    hyperparams = grid_cv_params or {
        "lr": 1e-4,
        "epochs": 30,
        "batch_size": 4
    }

    # Build vocabulary from all texts
    tokenizer_vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    reverse_vocab = {0: "<pad>", 1: "<unk>", 2: "<start>", 3: "<end>"}

    for text in texts:
        # DYNAMIC REPLACEMENT: Detect language per individual text chunk in the shared room data
        lang = detect_lang(text)
        tokens = tokenize(text, lang)
        
        for t in tokens:
            for w in [t['stem'], t['original']]:
                if w not in tokenizer_vocab:
                    tokenizer_vocab[w] = len(tokenizer_vocab)
                    reverse_vocab[tokenizer_vocab[w]] = w

    # Encode all documents
    encoded_docs = []
    for text in texts:
        # DYNAMIC REPLACEMENT: Detect language consistently during document mapping
        lang = detect_lang(text)
        tokens = tokenize(text, lang)
        
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
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]
    print(f"Starting training Run [{batch_version}] | Vocab Size: {len(tokenizer_vocab)} | H-Params: {hyperparams}")

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(encoded_docs), batch_size):
            batch = encoded_docs[i:i + batch_size]
            if not batch:
                continue

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
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Batch {batch_version} | Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # 2. SAVE VERSIONED WEIGHTS & SYNCED METADATA
    model_filename = f'companion_model_{batch_version}.pth'
    vocab_filename = f'tokenizer_vocab_{batch_version}.json'
    
    torch.save(model.state_dict(), model_filename)
    
    metadata = {
        "batch_version": batch_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "unique_words_trained": len(encoded_docs),
        "final_loss": avg_loss,
        "hyperparameters": hyperparams,
        "vocab": tokenizer_vocab, 
        "reverse": reverse_vocab
    }
    
    with open(vocab_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Overwrite active production assets
    torch.save(model.state_dict(), 'companion_model.pth')
    with open('tokenizer_vocab.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Deployment artifacts saved. Active production updated to: {batch_version}")


async def auto_train_monitor():
    """Background task that watches for new words in memory grid and retrains."""
    last_count = get_unique_words_count()
    print(f"Auto-train monitor started. Unique words: {last_count}")

    while True:
        await asyncio.sleep(60)
        current_count = get_unique_words_count()
        if current_count > last_count:
            print(f"New words detected ({current_count} vs {last_count}). Starting training...")
            await asyncio.to_thread(train_model)
            last_count = current_count
        else:
            last_count = current_count


def start_background_training():
    """Run the monitor in the current event loop."""
    loop = asyncio.get_event_loop()
    loop.create_task(auto_train_monitor())
