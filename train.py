import argparse

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import AesopDataset
from model import ReLUModel

THRESHOLD = 1.5

def compute_layer_goodness(layer_outputs):
    return [torch.sigmoid(out.norm(dim=-1) - THRESHOLD) for out in layer_outputs]

def compute_classifier_goodness(log_probs, target):
    return log_probs[range(target.shape[0]), target]

def sample_pred(outputs, target):
    prob_dist = outputs.exp()
    if prob_dist.isnan().sum() > 0: breakpoint()
    prob_dist[range(target.shape[0]), target] = 0
    return torch.multinomial(prob_dist, 1)

class Trainer():
    def __init__(self, model, train_loader, val_loader, writer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader =  val_loader
        self.writer = writer

    def update_layer_weights(self, inputs, outputs, lr):
        batch_size = inputs[0].shape[0]
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            prob = torch.sigmoid(out.norm(dim=-1, keepdim=True) - THRESHOLD)
            dgdo = (1 - prob) * out
            dodw = inp
            if dgdo.isnan().sum() + dodw.isnan().sum() > 0: breakpoint()
            self.model.layers[i].weight += lr * dgdo.T @ dodw
            self.model.layers[i].bias += (lr * dgdo).mean(0)

    def update_classifier_weights(self, inputs, outputs, target, lr):
        batch_size = inputs.shape[0]
        dgdo = -outputs.exp()
        dgdo[range(target.shape[0]), target] += 1
        dodw = inputs
        self.model.output_layer.weight += lr * dgdo.T @ dodw
        self.model.output_layer.bias += (lr * dgdo).mean(0)

    def train(self, train_args):
        step = 0

        for epoch in tqdm.tqdm(range(train_args.n_epochs), unit="epoch"):
            # Train for an epoch
            for i, (sample, target) in tqdm.tqdm(enumerate(self.train_loader)):
                step += 1
                with torch.no_grad(): # Not using backprop
                    # Positive step
                    positive_inputs, positive_outputs = self.model(sample)

                    # Negative step
                    pos_pred = sample_pred(positive_outputs[-1], target)
                    negative_sample = torch.cat([sample[:, 1:], pos_pred], dim=-1)
                    negative_inputs, negative_outputs = self.model(negative_sample)

                    # Update model
                    self.update_layer_weights(positive_inputs[:-1], positive_outputs[:-1], train_args.learning_rate)
                    self.update_layer_weights(negative_inputs[:-1], negative_outputs[:-1], -train_args.learning_rate)
                    self.update_classifier_weights(positive_inputs[-1], positive_outputs[-1], target, train_args.learning_rate)

                    # Logging
                    if step % train_args.log_every == 0:
                        classifier_goodness = compute_classifier_goodness(positive_outputs[-1], target)
                        self.writer.add_scalar("pos_goodness/classifier", classifier_goodness.mean(), step)

                        layer_goodness = compute_layer_goodness(positive_outputs[:-1])
                        for l, layer in enumerate(layer_goodness):
                            self.writer.add_scalar(f"pos_goodness/layer_{l}", layer.mean(), step)

                        layer_neg_goodness = compute_layer_goodness(negative_outputs[:-1])
                        for l, layer in enumerate(layer_neg_goodness):
                            self.writer.add_scalar(f"neg_goodness/layer_{l}", layer.mean(), step)

                        for l, (pos_layer, neg_layer) in enumerate(zip(layer_goodness, layer_neg_goodness)):
                            self.writer.add_scalar(f"net_goodness/layer_{l}", (pos_layer - neg_layer).mean(), step)

            # TODO: Evaluate on validation dataset, compute average perplexity of the next character


def train(args):
    # Load datasets
    dataset = AesopDataset(device=args.device)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    model = ReLUModel(
        vocab_dim=len(dataset.vocab),
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units
    ).to(args.device)
    writer = SummaryWriter("logs")

    step = 0
    trainer = Trainer(model, train_loader, val_loader, writer)
    trainer.train(args)
def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
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
