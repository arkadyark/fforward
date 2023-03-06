import argparse
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import AesopDataset
from model import ReLUModel

THRESHOLD = 1.5


def compute_layer_goodness(layer_outputs):
    return [torch.sigmoid(out.norm(dim=-1) - THRESHOLD) for out in layer_outputs]


def compute_classifier_goodness(log_probs, target):
    return log_probs[range(target.shape[0]), target]


def sample_pred(outputs, target):
    prob_dist = outputs.exp()
    prob_dist[range(target.shape[0]), target] = 0
    return torch.multinomial(prob_dist, 1)


class Trainer:
    def __init__(self, model, train_loader, val_loader, model_dir):
        self.model = model
        num_parameters = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {num_parameters} parameters")

        if (model_dir / "best_model.pt").exists():
            self.model = torch.load(
                model_dir / "best_model.pt", map_location=model.device
            )
            print("Loaded model from checkpoint")

        self.train_loader = train_loader
        self.val_loader = val_loader

        model_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir = model_dir

        self.train_writer = SummaryWriter(str(model_dir / "logs" / "train"))
        self.val_writer = SummaryWriter(str(model_dir / "logs" / "val"))

    def update_layer_weights(self, inputs, outputs, lr):
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            prob = torch.sigmoid(out.norm(dim=-1, keepdim=True) - THRESHOLD)
            dgdo = (1 - prob) * 2 * out
            dodw = inp
            self.model.layers[i].weight += lr * dgdo.T @ dodw
            self.model.layers[i].bias += (lr * dgdo).mean(0)

    def update_classifier_weights(self, inputs, outputs, target, lr):
        dgdo = -outputs.exp()
        dgdo[range(target.shape[0]), target] += 1
        dodw = inputs
        self.model.output_layer.weight += lr * dgdo.T @ dodw
        self.model.output_layer.bias += (lr * dgdo).mean(0)

    def get_accuracy(self, outputs, targets):
        return (outputs.argmax(-1) == targets).sum()

    def get_log_perplexity(self, outputs, targets):
        return outputs[range(len(targets)), targets].sum()

    def train_epoch(self, train_args):
        # Train for an epoch
        classifier_correct, log_perplexity, num_samples = 0, 0, 0
        self.model.train()
        for sample, target in tqdm.tqdm(self.train_loader):
            self.step += 1
            num_samples += sample.shape[0]
            with torch.no_grad():  # Not using backprop
                # Positive step
                positive_inputs, positive_outputs = self.model(sample)
                classifier_correct += self.get_accuracy(positive_outputs[-1], target)
                log_perplexity += self.get_log_perplexity(positive_outputs[-1], target)

                # Negative step
                pos_pred = sample_pred(positive_outputs[-1], target)
                negative_sample = torch.cat([sample[:, 1:], pos_pred], dim=-1)
                negative_inputs, negative_outputs = self.model(negative_sample)

                # Update model
                self.update_layer_weights(
                    positive_inputs[:-1],
                    positive_outputs[:-1],
                    train_args.learning_rate,
                )
                self.update_layer_weights(
                    negative_inputs[:-1],
                    negative_outputs[:-1],
                    -train_args.learning_rate,
                )
                self.update_classifier_weights(
                    positive_inputs[-1],
                    positive_outputs[-1],
                    target,
                    train_args.learning_rate,
                )

                # Logging
                if self.step % train_args.log_every == 0:
                    classifier_goodness = compute_classifier_goodness(
                        positive_outputs[-1], target
                    )
                    self.train_writer.add_scalar(
                        "pos_goodness/classifier", classifier_goodness.mean(), self.step
                    )

                    layer_goodness = compute_layer_goodness(positive_outputs[:-1])
                    for l, layer in enumerate(layer_goodness):
                        self.train_writer.add_scalar(
                            f"pos_goodness/layer_{l}", layer.mean(), self.step
                        )

                    layer_neg_goodness = compute_layer_goodness(negative_outputs[:-1])
                    for l, layer in enumerate(layer_neg_goodness):
                        self.train_writer.add_scalar(
                            f"neg_goodness/layer_{l}", layer.mean(), self.step
                        )

                    for l, (pos_layer, neg_layer) in enumerate(
                        zip(layer_goodness, layer_neg_goodness)
                    ):
                        self.train_writer.add_scalar(
                            f"net_goodness/layer_{l}",
                            (pos_layer - neg_layer).mean(),
                            self.step,
                        )

        accuracy = classifier_correct / num_samples
        self.train_writer.add_scalar(f"accuracy", accuracy, self.epoch)

        perplexity = torch.exp(-log_perplexity / num_samples)
        self.train_writer.add_scalar(f"perplexity", perplexity, self.epoch)

    def eval_epoch(self):
        self.model.eval()
        classifier_correct, log_perplexity, num_samples = 0, 0, 0
        for sample, target in tqdm.tqdm(self.val_loader):
            num_samples += sample.shape[0]
            _, positive_outputs = self.model(sample)
            classifier_correct += self.get_accuracy(positive_outputs[-1], target)
            log_perplexity += self.get_log_perplexity(positive_outputs[-1], target)

        accuracy = classifier_correct / num_samples
        self.val_writer.add_scalar(f"accuracy", accuracy, self.epoch)

        perplexity = torch.exp(-log_perplexity / num_samples)
        self.val_writer.add_scalar(f"perplexity", perplexity, self.epoch)

        return perplexity

    def train(self, train_args):
        self.step = 0
        best_perplexity = torch.inf

        for epoch in tqdm.tqdm(range(train_args.n_epochs), unit="epoch"):
            self.epoch = epoch
            self.train_epoch(train_args)

            # Save model, if necessary
            perplexity = self.eval_epoch()
            if perplexity < best_perplexity:
                torch.save(self.model, str(self.model_dir / "best_model.pt"))
                best_perplexity = perplexity


def train(args):
    # Load datasets
    dataset = AesopDataset(device=args.device)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    model = ReLUModel(
        vocab_dim=len(dataset.tokenizer.vocab),
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
    ).to(args.device)

    trainer = Trainer(model, train_loader, val_loader, args.model_dir)
    trainer.train(args)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=2000)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-dir", type=Path, default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    train(args)
