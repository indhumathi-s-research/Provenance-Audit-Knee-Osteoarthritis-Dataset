"""Training loop and evaluation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_metrics


def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches: int | None = None) -> tuple:
    model.train(train)
    running, correct, total = 0.0, 0, 0
    y_all, p_all = [], []
    pr = tqdm(loader, desc="train" if train else "val", leave=False, disable=not train)
    for bi, (x, y, *_ ) in enumerate(pr):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        running += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        y_all.append(y.detach().cpu().numpy())
        p_all.append(out.detach().cpu())
    y_all = np.concatenate(y_all)
    p_all = torch.cat(p_all).numpy()
    return {
        "loss": running / total,
        "accuracy": correct / total,
        "y": y_all,
        "proba": p_all,
    }


def train_model(
    cfg: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    run_dir,
) -> dict:
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"].get("early_stop_patience", 8)
    max_train_batches = cfg["training"].get("max_train_batches")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if cfg["training"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg["training"]["lr"], momentum=0.9,
            weight_decay=cfg["training"]["weight_decay"],
        )

    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_state = None
    patience_count = 0
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True, max_batches=max_train_batches)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        vmet = compute_metrics(va["y"], va["proba"].argmax(1), va["proba"])
        score = vmet["macro_f1"]
        if scheduler:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_acc": tr["accuracy"],
                "val_loss": va["loss"],
                "val_acc": vmet["accuracy"],
                "val_macro_f1": vmet["macro_f1"],
                "val_balanced_acc": vmet["balanced_accuracy"],
                "val_qwk": vmet["qwk"],
                "lr": scheduler.get_last_lr()[0] if scheduler else cfg["training"]["lr"],
            }
        )
        dt = time.perf_counter() - t_ep
        print(
            f"epoch {epoch:3d}/{epochs} ({dt:5.1f}s) | tr_loss {tr['loss']:.4f} tr_acc {tr['accuracy']:.4f} | "
            f"val_loss {va['loss']:.4f} | macroF1 {vmet['macro_f1']:.4f} | "
            f"bAcc {vmet['balanced_accuracy']:.4f} | QWK {vmet['qwk']:.4f}",
            flush=True,
        )
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"early stopping at epoch {epoch} (patience {patience})")
                break

    wall_s = time.perf_counter() - t_start
    return {
        "best_val_macro_f1": best_val,
        "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
        "epochs_run": len(history),
        "training_time_s": wall_s,
        "history": history,
        "best_state": best_state,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    y_all, p_all, meta = [], [], []
    with torch.no_grad():
        for x, y, path, pid, kid in tqdm(loader, desc="eval", disable=True):
            out = model(x.to(device))
            y_all.append(y.numpy())
            p_all.append(out.detach().cpu().numpy())
            for i in range(len(path)):
                meta.append({"image_path": path[i], "patient_id": pid[i], "knee_id": kid[i]})
    y = np.concatenate(y_all)
    p = np.concatenate(p_all)
    pred = p.argmax(1)
    m = compute_metrics(y, pred, p)
    df = pd.DataFrame(meta)
    df["true_label"] = y
    df["predicted_label"] = pred
    for i in range(p.shape[1]):
        df[f"probability_{i}"] = p[:, i]
    return m, df


def build_loaders(cfg: dict, split_df: pd.DataFrame, device: torch.device, seed: int):
    from torch.utils.data import DataLoader

    from src.dataset import KneeImageDataset, class_weights_from_df

    train_df = split_df[split_df["role"] == "train"]
    val_df = split_df[split_df["role"] == "val"]
    test_df = split_df[split_df["role"] == cfg["eval"]["test_role"]]

    train_ds = KneeImageDataset(train_df, cfg, train=True, cache_dir=cfg["preprocessing"].get("cache_dir"))
    val_ds = KneeImageDataset(val_df, cfg, train=False, cache_dir=cfg["preprocessing"].get("cache_dir"))
    test_ds = KneeImageDataset(test_df, cfg, train=False, cache_dir=cfg["preprocessing"].get("cache_dir"))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["workers"], generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["workers"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"]["workers"],
    )
    weights = class_weights_from_df(train_df, cfg["model"]["num_classes"])
    return train_loader, val_loader, test_loader, train_df, weights
