import argparse
import torch

class ReLUModel(torch.nn.Module):
    def __init__(self, input_len=10, hidden_layers=3, hidden_units=2000, vocab_dim=30):
        super().__init__()
        self.layers = [
            torch.nn.Linear(input_len if i == 0 else hidden_units, hidden_units)
            for i in range(hidden_layers)
        ]
        self.output_layer = torch.nn.Linear(hidden_layers * hidden_units, vocab_dim)

        for layer in self.layers: layer.weight.requires_grad = False
        self.output_layer.weight.requires_grad = False

    def forward(self, inputs):
        outs = []
        layer_input = inputs
        for layer in self.layers:
            out = torch.nn.functional.relu(layer(layer_input))
            outs.append(out)
            norm = out.norm(dim=-2)
            norm[norm == 0] = 1 # Prevent nans
            layer_input = out / norm
        output = self.output_layer(torch.cat(outs, dim=-1))
        return torch.softmax(output, dim=-1), (outs, output)

class AesopDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="aesop.txt"):
        self.vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ;,.")}
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
        string = [self.vocab[c] for c in " ".join(lines).lower() if c in self.vocab]
        return [
            (torch.tensor(string[i:i+10], dtype=torch.float32), string[i+10])
            for i in range(len(string) - 10)
        ]

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

def compute_norm_goodness(layer_output):
    return [l.norm() for l in layer_output]

def compute_prob(output, target):
    return 1/(1 + torch.exp(-output[:, target]))

def train_step(model, sample, last_output=None):
    # Positive training
    positive_output, positive_outputs = model(sample)

    # Negative training
    if last_output is not None:
        sample[-1] = last_output
        _, negative_outputs = model(sample)
    else:
        negative_outputs = None

    last_output = torch.argmax(positive_output)
    return positive_outputs, negative_outputs, last_output

def update_weights(model, outputs, target, lr, negative=False):
    if outputs is None: return
    layer_outputs, final_output = outputs
    for i, o in enumerate(layer_outputs):
        # For intermediate goodnesses, gradient is 2 * g
        gradient = 2 * o
        if negative: gradient = -gradient
        # Apply gradient to corresponding layer
        model.layers[i].weight += lr * (model.layers[i].weight.T * gradient).T
        model.layers[i].bias += lr * gradient[0]

    # For final output, log probability of correct output
    gradient = -1/(1 + torch.exp(final_output[:, target]))
    if negative: gradient = -gradient
    model.output_layer.weight += lr * (model.output_layer.weight.T * gradient).T
    model.output_layer.bias += lr * gradient[0]

def compute_goodness(outputs, target):
    if outputs is None: return 0
    layer_outputs, final_output = outputs
    layer_goodnesses = [layer.norm() for layer in layer_outputs]
    final_goodness = torch.log(1 / (1 + torch.exp(-final_output[:, target])))
    return layer_goodnesses, final_goodness

def train(args):
    dataset = AesopDataset()
    model = ReLUModel(
        vocab_dim=len(dataset.vocab),
        hidden_layers=3,
        hidden_units=2000
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for epoch in range(args.n_epochs):
        samples_seen = 0
        last_output = None
        for i, (sample, target) in enumerate(data_loader):
            if target < 0: 
                last_output = None
                continue
            with torch.no_grad():
                positive_outputs, negative_outputs, last_output = train_step(model, sample, last_output)
                print(f"Positive goodness for step {i}: {compute_goodness(positive_outputs, target)}")
                print(f"Negative goodness for step {i}: {compute_goodness(negative_outputs, target)}")
                update_weights(model, positive_outputs, target, args.learning_rate)
                update_weights(model, negative_outputs, target, args.learning_rate, negative=True)

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = load_args()
    train(args)
