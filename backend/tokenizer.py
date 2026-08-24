import re

class CodeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        self.build_vocab([])

    def build_vocab(self, texts):
        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                self.reverse_vocab[self.vocab[tok]] = tok
        for text in texts:
            for t in self.tokenize_text(text):
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                    self.reverse_vocab[self.vocab[t]] = t

    def tokenize_text(self, text):
        pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[{}()\[\].=+*/<>!&|;,\-\"']"
        tokens = re.findall(pattern, text.lower())
        return [self.special_tokens[2]] + tokens + [self.special_tokens[3]]

    def encode(self, text):
        tokens = self.tokenize_text(text)
        return [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]

    def decode(self, ids):
        out = []
        for id in ids:
            tok = self.reverse_vocab.get(id, '<unk>')
            if tok in self.special_tokens:
                continue
            out.append(tok)
        return ' '.join(out)

    def vocab_size(self):
        return len(self.vocab)