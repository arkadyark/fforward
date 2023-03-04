def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-dir", type=Path, default="exp")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args

def generate(args):
    model = torch.load(model, map_location=args.device)


if __name__ == "__main__":
    args = load_args()
    generate(args)
