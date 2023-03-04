import torch
from tokenizer import Tokenizer

class AesopDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="aesop.txt", device="cpu"):
        self.tokenizer = Tokenizer()
        self.device = device
        self.samples = []
        with open(data_path) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line.isupper():
                    if len(lines) > 0:
                        self.samples += self._make_samples(lines)
                        lines = []
                    continue
                lines.append(line)
            self.samples += self._make_samples(lines)

    def _make_samples(self, lines):
        fable = " ".join(lines).lower()
        string = self.tokenizer.encode(fable)
        return [
            (torch.tensor(string[i:i+10], device=self.device),
             torch.tensor(string[i+10], device=self.device))
            for i in range(len(string) - 10)
        ]

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)
