import json
import argparse
import torch
from models.model import FishBehaviorModel
from data.dataset import prepare_data
from training.trainer import train_model

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.json", help="Path to JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")

    train_paths, train_labels, val_paths, val_labels = prepare_data(config)
    model = FishBehaviorModel(config).to(device)
    train_model(model, train_paths, train_labels, val_paths, val_labels, config, device)

if __name__ == "__main__":
    main()
