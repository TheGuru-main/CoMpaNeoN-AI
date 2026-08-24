import os
import random

class DataMixer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.texts = []
        self.load()

    def load(self):
        self.texts = []
        if os.path.isdir(self.data_dir):
            for fname in os.listdir(self.data_dir):
                if fname.endswith('.txt'):
                    path = os.path.join(self.data_dir, fname)
                    with open(path, 'r', encoding='utf-8') as f:
                        self.texts.extend([line.strip() for line in f if line.strip()])

    def get_batch(self, batch_size):
        return random.sample(self.texts, min(batch_size, len(self.texts)))