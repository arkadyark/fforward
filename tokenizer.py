class Tokenizer:
    VOCAB = "abcdefghijklmnopqrstuvwxyz ;,."

    def __init__(self):
        self.vocab = {c: i for i, c in enumerate(self.VOCAB)}
        self.i2c = {i: c for i, c in enumerate(self.VOCAB)}

    def encode(self, text):
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, tokens):
        return "".join([self.i2c.get(c, "") for c in tokens])
