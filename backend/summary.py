"""
Summary Module

Generates a concise, AI-aware summary of a workspace conversation.
Uses the same transformer and context retrieval.
"""

import os
import json
import torch
import torch.nn.functional as F
from ai_model import MiniCompanionAI
from tokenizer import tokenize, normalize_lang
from memory_grid import MemoryGrid
from word_understanding import WordUnderstanding
from prompts_manager import build_prompt

# Load tokenizer vocab and model if available
def load_model_and_vocab():
    vocab = None
    reverse = None
    model = None
    if os.path.exists('tokenizer_vocab.json'):
        with open('tokenizer_vocab.json', 'r') as f:
            data = json.load(f)
            vocab = data['vocab']
            reverse = {int(k): v for k, v in data['reverse'].items()}
    if os.path.exists('companion_model.pth'):
        model = MiniCompanionAI(len(vocab) if vocab else 1000)
        model.load_state_dict(torch.load('companion_model.pth', map_location='cpu'))
        model.eval()
    return model, vocab, reverse

def encode_text(text, vocab):
    tokens = tokenize(text, "en")
    ids = [vocab.get("<start>", 2)]
    for t in tokens:
        word = t['stem'] if t['stem'] in vocab else (t['original'] if t['original'] in vocab else "<unk>")
        ids.append(vocab.get(word, 1))
    ids.append(vocab.get("<end>", 3))
    return ids

def decode_ids(ids, reverse):
    return ' '.join([reverse.get(i, '<unk>') for i in ids if i not in {0,2,3}])

def generate_summary(conversation_text, max_len=100, temperature=0.7):
    """
    Generate a summary using the trained model.
    If model is not available, return first 500 chars as fallback.
    """
    model, vocab, reverse = load_model_and_vocab()
    if model is not None and vocab is not None:
        # Prepare prompt: ask for summary
        prompt = f"Summarize the following conversation briefly:\n{conversation_text}\nSummary:"
        input_ids = torch.tensor([encode_text(prompt, vocab)], dtype=torch.long)
        output_ids = []
        with torch.no_grad():
            for _ in range(max_len):
                logits = model(input_ids)[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                output_ids.append(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
                if next_id == vocab.get("<end>", 3):
                    break
        summary = decode_ids(output_ids, reverse)
        return summary
    else:
        # fallback: truncated text
        return conversation_text[:500] + ("..." if len(conversation_text) > 500 else "")