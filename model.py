import torch
import torch.nn.functional as F

EPSILON = 1e-12

class ReLUModel(torch.nn.Module):
    def __init__(self, input_len=10, hidden_layers=3, hidden_units=2000, vocab_dim=30):
        super().__init__()
        self.input_len = input_len
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
            inp = layer_input / (layer_input.norm(dim=-1, keepdim=True) + EPSILON)
            ins.append(inp)
            layer_input = F.relu(layer(inp))
            outs.append(layer_input)

        # Final layer
        layer_input = torch.cat(outs, dim=-1)
        ins.append(layer_input)
        inp = torch.cat(outs, dim=-1)
        probs = torch.log_softmax(self.output_layer(inp), -1)
        outs.append(probs)

        return ins, outs

    def generate(self, prompt, max_len=100):
        prompt_suffix = prompt[-input_len:]
        output = ''
        # TODO - fill this in

