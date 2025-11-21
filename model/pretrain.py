#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrain script (no model-file change required).

Features:
- Grid-sweep over (tau, tau_phys_scale)
- For each (tau, scale) train 4 models with property_index = 0, 1, 2, None (in that order)
- Runtime monkey-patch to log per-step tau_phys (from _soft_labels_from_distance)
- Per-run artifacts: config.json, history.csv, tau_phys_epoch.csv, best_model.pth, training_curve.png
- Global sweep_summary.csv collected at base_save_dir

Author: (your name)
"""

import os
import json
import csv
import time
from typing import Dict, Any, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import MethodType

from torch.utils.data import DataLoader as TorchDataLoader, random_split

from model.getdata import smiles2graph
from model.mpn_vas_st import GNNModelWithNewLoss

class FilterEmptyGraphs(torch.utils.data.Dataset):
    """Filter out graphs with no nodes or no edges."""
    def __init__(self, dataset):
        self.filtered_dataset = [d for d in dataset
                                 if getattr(d, "x", None) is not None and d.x.size(0) > 0
                                 and getattr(d, "edge_index", None) is not None and d.edge_index.size(1) > 0]
    def __len__(self): return len(self.filtered_dataset)
    def __getitem__(self, i): return self.filtered_dataset[i]


def read_vsa_data(vsa_file: str):
    """
    Expect CSV with columns: SMILES, SMR_VSA, SlogP_VSA, PEOE_VSA
    Where each VSA field looks like: "[1.0 2.0 3.0 ...]"
    """
    df = pd.read_csv(vsa_file)

    def parse_vsa(s: str):
        try:
            return list(map(float, s.strip('[]').split()))
        except Exception:
            return []

    smr_arrays   = df["SMR_VSA"].apply(parse_vsa).tolist()
    slogp_arrays = df["SlogP_VSA"].apply(parse_vsa).tolist()
    peoe_arrays  = df["PEOE_VSA"].apply(parse_vsa).tolist()

    properties = list(zip(smr_arrays, slogp_arrays, peoe_arrays))
    return df["SMILES"].tolist(), properties


# =========================
# Monkey-patch: enable tau_phys logging (no model-file change)
# =========================
def enable_tau_logging(model):
    """
    Patch model at runtime:
      - Wrap _soft_labels_from_distance to capture returned tau_val each step
      - Provide begin_epoch() / end_epoch_stats() hooks to collect epoch stats
    """
    if not hasattr(model.__class__, "_soft_labels_from_distance"):
        print("[warn] model class has no _soft_labels_from_distance; tau_phys logging disabled.")
        return

    model._epoch_tau_phys_samples = []
    model._tau_phys_epoch_stats   = []

    # Unbound original method from the class
    orig_unbound = model.__class__._soft_labels_from_distance

    def _wrapped_soft_labels_from_distance(self, D, *args, **kwargs):
        Y, tau_val = orig_unbound(self, D, *args, **kwargs)
        # tau_val may be a tensor
        try:
            v = float(getattr(tau_val, "detach", lambda: tau_val)().item())
        except Exception:
            v = float(tau_val)
        self._epoch_tau_phys_samples.append(v)
        return Y, tau_val

    # Bind to instance
    model._soft_labels_from_distance = MethodType(_wrapped_soft_labels_from_distance, model)

    # Epoch hooks
    def _begin_epoch(self):
        self._epoch_tau_phys_samples = []

    def _end_epoch_stats(self) -> Dict[str, float]:
        arr = self._epoch_tau_phys_samples or []
        if not arr:
            stats = dict(
                tau_phys_mean=float("nan"),
                tau_phys_median=float("nan"),
                tau_phys_min=float("nan"),
                tau_phys_max=float("nan"),
                tau_phys_count=0.0,
            )
        else:
            t = torch.tensor(arr, dtype=torch.float32)
            stats = dict(
                tau_phys_mean=float(t.mean().item()),
                tau_phys_median=float(t.median().item()),
                tau_phys_min=float(t.min().item()),
                tau_phys_max=float(t.max().item()),
                tau_phys_count=float(t.numel()),
            )
        model._tau_phys_epoch_stats.append(stats)
        return stats

    model.begin_epoch     = MethodType(_begin_epoch, model)
    model.end_epoch_stats = MethodType(_end_epoch_stats, model)


# =========================
# Training loop (drop-in replacement; no model-file changes)
# =========================
def train_with_list_batches(
    model,
    data_list,                      # List[Data]
    save_path: str = "models_mpn",
    num_epochs: int = 1000,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    patience: int = 50,
    batch_size: int = 128,
    best_val_loss_all: float = float('inf'),
    exp_name: Optional[str] = None,
) -> float:
    """
    Train one model instance on (train,val) split; save artifacts to save_path.
    Returns best validation loss.
    """
    print(f"[train] Saving artifacts to: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Persist one-off config (includes tau & tau_phys_scale)
    config = {
        "exp_name": exp_name or "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": num_epochs, "lr": lr, "weight_decay": weight_decay,
        "patience": patience, "batch_size": batch_size,
        "tau": float(getattr(model, "tau", 0.1)),
        "tau_phys": None if getattr(model, "tau_phys", None) is None else float(model.tau_phys),
        "tau_phys_scale": float(getattr(model, "tau_phys_scale", 0.5)),
        "property_index": getattr(model, "property_index", None),
        "hidden_dim": getattr(model, "hidden_dim", None),
        "dropout_rate": getattr(model, "dropout_rate", None),
        "cov_num": getattr(model, "cov_num", None),
    }
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Dataset & loaders
    ds = FilterEmptyGraphs(data_list)
    n_train = int(0.8 * len(ds))
    train_set, val_set = random_split(ds, [n_train, len(ds) - n_train])

    collate = lambda xs: xs  # keep list[Data]
    train_loader = TorchDataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   drop_last=True, collate_fn=collate)
    val_loader   = TorchDataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                   drop_last=False, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(patience // 2, 1))

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    # CSV logs
    hist_csv = os.path.join(save_path, "history.csv")
    tau_csv  = os.path.join(save_path, "tau_phys_epoch.csv")
    with open(hist_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])
    with open(tau_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "tau_phys_mean", "tau_phys_median",
                                "tau_phys_min", "tau_phys_max", "tau_phys_count"])

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        if hasattr(model, "begin_epoch"):
            model.begin_epoch()

        # Train
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            if (not torch.isfinite(loss)) or (not loss.requires_grad):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1

        avg_train_loss = total_loss / max(steps, 1)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        val_loss, vsteps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                l = model.get_loss(batch)
                if torch.isfinite(l):
                    val_loss += float(l.item())
                    vsteps += 1
        avg_val_loss = val_loss / max(vsteps, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # Epoch tau_phys stats
        if hasattr(model, "end_epoch_stats"):
            stats = model.end_epoch_stats()
        else:
            stats = dict(tau_phys_mean=float("nan"), tau_phys_median=float("nan"),
                         tau_phys_min=float("nan"), tau_phys_max=float("nan"),
                         tau_phys_count=0.0)

        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, lr_now])
        with open(tau_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, stats["tau_phys_mean"], stats["tau_phys_median"],
                                    stats["tau_phys_min"], stats["tau_phys_max"], stats["tau_phys_count"]])

        print(f"Epoch {epoch}/{num_epochs} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
              f"LR {lr_now:.2e} | tau_phys(mean) {stats['tau_phys_mean']:.4g}")

        # Save best
        if avg_val_loss < best_val_loss and avg_val_loss < best_val_loss_all:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'encoder_state_dict': model.state_dict(),
                'config': config,
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Plot with baselines (log N)
    def avg_logN(dloader):
        Ns = [max(len(batch) - 1, 1) for batch in dloader]
        return float(np.log(np.mean(Ns))) if Ns else 0.0

    train_eval_loader = TorchDataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    val_eval_loader   = TorchDataLoader(val_set,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    baseline_train = avg_logN(train_eval_loader)
    baseline_val   = avg_logN(val_eval_loader)

    if val_losses:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses,   label='Val Loss')
        plt.axhline(y=baseline_train, linestyle='--', linewidth=1.5,
                    label=f'Train Baseline (log N)≈{baseline_train:.3f}')
        plt.axhline(y=baseline_val,   linestyle=':',  linewidth=1.5,
                    label=f'Val Baseline (log N)≈{baseline_val:.3f}')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'Training Process (Best Val Loss: {best_val_loss:.4f})')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(save_path, "training_curve.png"))
        plt.close()

    return best_val_loss


# =========================
# Sweep controller: For each (tau, scale), train 4 models (0,1,2,None)
# =========================
def sweep_tau_four_tasks(
    base_save_dir: str,
    build_model_for_task,     # fn(property_index: Optional[int]) -> untrained model
    data_list,
    taus: Iterable[float],
    tau_phys_scales: Iterable[float],
    train_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[list, str]:
    """
    For each (tau, tau_phys_scale), train 4 models in fixed order:
      0 -> 1 -> 2 -> None
    Directory layout:
      base_save_dir/
        tau_<tau>__scale_<s>/
          task_0/ ... task_1/ ... task_2/ ... task_none/ ...
    Returns: (results, summary_csv_path)
    """
    os.makedirs(base_save_dir, exist_ok=True)
    results = []  # (tau, scale, task_name, property_index, best_val_loss)

    task_specs = [
        ("task_0", int(0)),      # property_index = 0
        ("task_1", int(1)),      # property_index = 1
        ("task_2", int(2)),      # property_index = 2
        ("task_none", None) # fingerprint path; property_index = None
    ]

    for tau in taus:
        for s in tau_phys_scales:
            combo_dir = os.path.join(base_save_dir, f"tau_{tau:g}__scale_{s:g}")
            os.makedirs(combo_dir, exist_ok=True)

            for task_name, pidx in task_specs:
                # Build a fresh model for this task
                model = build_model_for_task(pidx)
                # Override sweep hyperparams
                model.tau = float(tau)
                model.tau_phys_scale = float(s)
                model = model.to(model.device)

                # Enable tau logging (monkey-patch)
                enable_tau_logging(model)

                # Run-specific directory
                save_dir = os.path.join(combo_dir, task_name)
                os.makedirs(save_dir, exist_ok=True)

                # Train
                kw = dict(num_epochs=1000, lr=5e-5, weight_decay=1e-4,
                          patience=50, batch_size=512, best_val_loss_all=float('inf'))
                if train_kwargs:
                    kw.update(train_kwargs)

                best = train_with_list_batches(
                    model, data_list, save_path=save_dir,
                    exp_name=f"tau={tau},scale={s},{task_name}", **kw
                )
                results.append((tau, s, task_name, pidx, best))

    # Summary CSV
    summary_csv = os.path.join(base_save_dir, "sweep_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "tau_phys_scale", "task", "property_index", "best_val_loss"])
        for tau, s, tname, pidx, best in results:
            w.writerow([tau, s, tname, "None" if pidx is None else pidx, best])

    print("[sweep] Finished. Summary saved to:", summary_csv)
    return results, summary_csv


# =========================
# Main
# =========================
def main():
    # ---- 0) config ----
    vsa_csv_path  = "./data/vsa_zinc_25k.csv"# vsa_zinc_50k.csv"
    base_save_dir = "mpnn_st_50_t/3/sweep"   # root output dir

    taus = (0.05, 0.1, 0.2, 0.4)             # sweep list for logits temperature
    tau_phys_scales = (0.25, 0.5, 0.75, 1.0) # sweep list for physical scale

    default_train_kwargs = dict(
        num_epochs=1,
        lr=5e-5,
        weight_decay=1e-4,
        patience=50,
        batch_size=512,
        best_val_loss_all=float('inf'),
    )

    # ---- 1) Load data ----
    x_smiles, properties = read_vsa_data(vsa_csv_path)
    data_list = smiles2graph(x_smiles, properties=properties)
    print("[data] Example Data object:", data_list[0])

    atom_dim = data_list[0].x.shape[1]
    bond_dim = data_list[0].edge_attr.shape[1]
    # num_global_features may be 0; you can read if present:
    # glob_dim = getattr(data_list[0], 'global_features', torch.zeros(1, 0)).shape[1]

    # ---- 2) Model factory: returns a NEW, untrained model for a given property_index ----
    def build_model_for_task(property_index: Optional[int]):
        m = GNNModelWithNewLoss(
            num_node_features=atom_dim,
            num_edge_features=bond_dim,
            num_global_features=0,
            hidden_dim=512, cov_num=3, dropout_rate=0.1,
            property_index=property_index,   # 0, 1, 2, or None
            save_path="models_mpn"
        )
        # If your project requires add_task, keep it consistent with your previous usage:
        if hasattr(m, "add_task"):
            m.add_task(property_index=property_index, save_path="models_mpn")
        return m

    # ---- 3) Sweep over (tau, tau_phys_scale) and train FOUR models per pair ----
    sweep_tau_four_tasks(
        base_save_dir=base_save_dir,
        build_model_for_task=build_model_for_task,
        data_list=data_list,
        taus=taus,
        tau_phys_scales=tau_phys_scales,
        train_kwargs=default_train_kwargs
    )


if __name__ == "__main__":
    main()
