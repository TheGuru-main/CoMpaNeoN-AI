import os
import random

class DataMixer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.texts = []
        self.load()

    def load(self):
        self.texts = []
        for fname in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, fname)
            if os.path.isfile(path) and fname.endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                    self.texts.extend([line for line in lines if line.strip()])

    def get_batch(self, batch_size):
        return random.sample(self.texts, min(batch_size, len(self.texts)))