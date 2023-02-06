import argparse

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

HIDDEN_UNITS = 2000
THRESHOLD = 2

class ReLUModel(torch.nn.Module):
    def __init__(self, input_len=10, hidden_layers=3, hidden_units=HIDDEN_UNITS, vocab_dim=30):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_len * vocab_dim if i == 0 else hidden_units, hidden_units)
            for i in range(hidden_layers)
        ])
        self.output_layer = torch.nn.Linear(hidden_layers * hidden_units, vocab_dim)

        for layer in self.layers: layer.weight.requires_grad = False
        self.output_layer.weight.requires_grad = False

    def forward(self, inputs):
        outs = []
        ins = []
        layer_input = F.one_hot(inputs, self.vocab_dim).flatten(1, -1).float()

        # Linear layers
        for layer in self.layers:
            inp = layer_input / layer_input.norm()
            ins.append(inp)
            layer_input = F.relu(layer(inp))
            outs.append(layer_input)

        # Final layer
        layer_input = torch.cat(outs, dim=-1)
        ins.append(layer_input)
        inp = torch.cat(outs, dim=-1)
        probs = torch.softmax(self.output_layer(inp), -1)
        outs.append(probs)

        return ins, outs

class AesopDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="aesop.txt", device="cpu"):
        self.vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ;,.")}
        self.device = device
        self.samples = []
        with open(data_path) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line.isupper():
                    if len(lines) > 0:
                        self.samples += self._make_samples(lines)
                        self.samples.append((-1, -1))
                        lines = []
                    continue
                lines.append(line)
            self.samples += self._make_samples(lines)

    def _make_samples(self, lines):
        fable = " ".join(lines).lower()
        string = [self.vocab[c] for c in fable if c in self.vocab]
        return [
            (torch.tensor(string[i:i+10], device=self.device),
             torch.tensor(string[i+10], device=self.device))
            for i in range(len(string) - 10)
        ]

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

def update_weights(model, inputs, outputs, target, lr):
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        if i == len(inputs) - 1:
            if lr < 0:
                # Do not update classifier layer on negative data
                continue
            dgdo = -out.clone()
            dgdo[:, target] += 1
            layer = model.output_layer
        else:
            prob = torch.sigmoid(out.norm() - THRESHOLD)
            dgdo = (1 - prob) * out
            layer = model.layers[i]
        dodw = inp
        dodb = 1 # Just for completeness
        layer.weight += lr * dgdo.T * dodw
        layer.bias += (lr * dgdo * dodb).squeeze()

def compute_goodness(outputs, target):
    layer_outs, probs = outputs[:-1], outputs[-1]
    layer_goodnesses = [torch.sigmoid(out.norm() - THRESHOLD) for out in layer_outs]
    classifier_goodness = torch.log(probs[:, target])
    return layer_goodnesses, classifier_goodness

def sample_last_output(sample, outputs):
    outputs[:, outputs.argmax(-1)] = 0
    return torch.multinomial(outputs, 1)

def train(args):
    dataset = AesopDataset(device=args.device)
    model = ReLUModel(
        vocab_dim=len(dataset.vocab),
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units
    ).to(args.device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    writer = SummaryWriter("logs")

    step = 0
    for epoch in tqdm.tqdm(range(args.n_epochs), unit="epoch"):
        last_output = None
        for i, (sample, target) in tqdm.tqdm(enumerate(data_loader)):
            step += 1
            if target < 0: 
                # Reset last output once we reach the end of a story
                last_output = None
                continue
            with torch.no_grad():
                # Positive step
                positive_inputs, positive_outputs = model(sample)
                layer_goodness, classifier_goodness = compute_goodness(positive_outputs, target)
                if not step % args.log_every:
                    writer.add_scalar("pos_goodness/classifier", classifier_goodness, step)
                    for l, layer in enumerate(layer_goodness):
                        writer.add_scalar(f"pos_goodness/layer_{l}", layer, step)

                # Negative step
                if last_output is not None:
                    sample[:, -1] = last_output
                    negative_inputs, negative_outputs = model(sample)
                    layer_neg_goodness, classifier_neg_goodness = compute_goodness(negative_outputs, target)
                    if not step % args.log_every:
                        writer.add_scalar("neg_goodness/classifier", classifier_neg_goodness, step)
                        for l, layer in enumerate(layer_neg_goodness):
                            writer.add_scalar(f"neg_goodness/layer_{l}", layer, step)
                        writer.add_scalar("net_goodness/classifier", classifier_goodness - classifier_neg_goodness, step)
                        for l, (pos_layer, neg_layer) in enumerate(zip(layer_goodness, layer_neg_goodness)):
                            writer.add_scalar(f"net_goodness/layer_{l}", pos_layer - neg_layer, step)

                update_weights(model, positive_inputs, positive_outputs, target, args.learning_rate)
                if last_output is not None:
                    update_weights(model, negative_inputs, negative_outputs, target, -args.learning_rate)

                last_output = sample_last_output(target, positive_outputs[-1])

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=2000)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = load_args()
    train(args)
