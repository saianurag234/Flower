import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss


def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
         device: torch.device):
    model.eval()

    test_loss, test_acc = 0, 0
    y_pred = []
    y_true = []
    y_proba = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = loss_fn(output.view(-1), target.float())

            test_loss += loss.item()

            _, predicted = torch.max(output, 1)
            probabilities = torch.sigmoid(output)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(target.cpu().numpy())
            y_proba.extend(probabilities.cpu().numpy())

    test_acc = accuracy_score(y_true, y_pred)

    test_loss /= len(dataloader)

    return test_loss, test_acc * 100, np.array(y_pred), np.array(y_true)


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = loss_fn(output.view(-1), target.float())
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(output, 1)
        train_acc += (predicted == target).sum().item() / len(target)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc * 100


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
          epochs: int, device: torch.device) -> Dict[str, List]:

    results = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), colour="BLUE"):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                           optimizer=optimizer, device=device)

        val_loss, val_acc, _, _ = test(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        print(
            f"\tTrain Epoch: {epoch + 1} \t"
            f"Train_loss: {train_loss:.4f} | "
            f"Train_acc: {train_acc:.4f} % | "
            f"Validation_loss: {val_loss:.4f} | "
            f"Validation_acc: {val_acc:.4f} %"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results
