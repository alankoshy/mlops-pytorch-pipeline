import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataset import get_dataloaders
from src.model import build_model


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def train():
    config_path = os.environ.get("CONFIG_PATH", os.path.join("configs", "training_config.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    cfg_train = config.get("training", {})
    cfg_data = config.get("data", {})
    cfg_model = config.get("model", {})
    cfg_output = config.get("output", {})

    device = torch.device(cfg_train.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        data_dir=cfg_data.get("data_dir", "./data"),
        batch_size=cfg_train.get("batch_size", 64),
        num_workers=cfg_data.get("num_workers", 2),
    )

    model = build_model(
        num_classes=cfg_model.get("num_classes", 10),
        pretrained=cfg_model.get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.get("learning_rate", 0.01))
    
    # Read patience directly from cfg_train
    early_stopper = EarlyStopping(
        patience=cfg_train.get("early_stopping_patience", 3),
        min_delta=cfg_train.get("early_stopping_min_delta", 0.001),
    )

    # Read checkpoint paths from cfg_output
    checkpoint_dir = cfg_output.get("checkpoint_dir", "./checkpoints")
    checkpoint_file = cfg_output.get("model_name", "best_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    best_val_loss = float("inf")

    for epoch in range(1, cfg_train["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, checkpoint_path)

        if early_stopper.step(val_loss):
            print(json.dumps({"event": "early_stopping_triggered", "epoch": epoch}))
            break


if __name__ == "__main__":
    train()
