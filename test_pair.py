#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for pairwise solvation regression models (solute + solvent + temperature).

This script loads:
  - a trained pairwise solvation model checkpoint (state_dict),
  - the corresponding ChemSpace encoder checkpoints,
and evaluates the model on one or multiple external solvation datasets.

Each test dataset is expected to contain the following columns
(case-insensitive, with simple fallbacks):
  - solute_smiles
  - solvent_smiles
  - temperature  (or 't')
  - logS         (or 'logs')

For each test set, the script reports:
  - MSE (Mean Squared Error)
  - pct(|pred - logS| <= 1.0)   # coverage within 1 log unit

Typical usage
-------------
Example:

    python test_pair_solvation.py \\
        --train_csv datasets/bigsoldb_chemprop_nonaq.csv \\
        --test_csvs datasets/leeds_all_chemprop.csv,\\
                    datasets/solprop_chemprop_nonaq.csv \\
        --encoder_ckpts "ckpt/tau_0.1__scale_0.25/task_0/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_1/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_2/best_model.pth,\\
                         ckpt/tau_0.1__scale_0.25/task_none/best_model.pth" \\
        --space_indices 0,1,2 \\
        --model_ckpt ckpt/leeds.pth \\
        --batch_size 64 \\
        --emb_dim 512 \\
        --mlp_hidden 1024,128,64 \\
        --t_embed_dim 32 \\
        --out_dir results/solvation_pair_test

Notes
-----
- The `train_csv` is only used to build sample graphs in order to infer
  node/edge feature dimensions; no training is performed.
- The model checkpoint is assumed to contain either:
    * a dict with key 'model_state_dict', or
    * a plain state_dict that can be loaded directly.
"""

import os
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

# ---------- external project modules ----------
from model.getdata import smiles2graph
from model.mpn_vas_st import GNNModelWithNewLoss
from model.fusion import WeightedFusion

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[info] Using device:", device)

# -------------------------------------------------------------------------
# Global hyperparameters (only EMB_DIM matters for building encoders)
# -------------------------------------------------------------------------
EMB_DIM = 512  # can be overridden by CLI


# ============================ basic utils ============================

def ensure_dir(p: str):
    """Create directory `p` if it does not exist."""
    os.makedirs(p, exist_ok=True)


def pct_within_1_np(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute percentage of samples satisfying |pred - label| <= 1.

    Parameters
    ----------
    preds : np.ndarray
        Model predictions.
    labels : np.ndarray
        Ground-truth labels.

    Returns
    -------
    float
        Fraction of samples with absolute error <= 1.
    """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    err = np.abs(preds - labels)
    return float((err <= 1.0).mean())


# ================== CSV → graphs ==================

def _resolve_col(df: pd.DataFrame, name: str, fallback: Optional[List[str]] = None) -> str:
    """
    Resolve a column name in a DataFrame in a case-insensitive way.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    name : str
        Desired column name (case-insensitive).
    fallback : list of str, optional
        Additional candidate names to try if `name` is not found.

    Returns
    -------
    str
        The actual column name in `df`.

    Raises
    ------
    KeyError
        If no matching column is found.
    """
    norm = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key in norm:
        return norm[key]
    if fallback:
        for cand in fallback:
            cand_key = cand.strip().lower()
            if cand_key in norm:
                return norm[cand_key]
    raise KeyError(f"Column {name} not found in {df.columns}")


def _smiles_graph_inputs_from_csv(csv_path: str):
    """
    Read a solvation CSV and convert solute / solvent SMILES to graph objects.

    Expected columns (case-insensitive):
        - solute_smiles
        - solvent_smiles
        - temperature (or 't')
        - logS (or 'logs')

    Returns
    -------
    d1, d2 : list of Data
        Graphs for solute and solvent.
    y : np.ndarray
        logS labels.
    T : np.ndarray
        Temperature values.
    """
    df = pd.read_csv(csv_path)

    solute_col = _resolve_col(df, "solute_smiles", ["solute_smiles"])
    solvent_col = _resolve_col(df, "solvent_smiles", ["solvent_smiles"])
    temp_col = _resolve_col(df, "temperature", ["temperature", "t"])
    label_col = _resolve_col(df, "logS", ["logs", "logS"])

    y = df[label_col].astype(float).to_numpy()
    T = df[temp_col].astype(float).to_numpy()

    solute = df[solute_col].astype(str).tolist()
    solvent = df[solvent_col].astype(str).tolist()

    d1 = smiles2graph(solute, y=y, properties=T)
    d2 = smiles2graph(solvent, y=y, properties=T)
    return d1, d2, y, T


# ================== dataset classes & loaders ==================

class PairDataset(Dataset):
    """
    Dataset over solute / solvent pairs, temperature, and label.

    Each item is:
        (solute_graph, solvent_graph, temperature, label)
    """
    def __init__(self, s, v, temps, labels):
        self.s = s
        self.v = v
        self.temps = temps
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.s[idx], self.v[idx], float(self.temps[idx]), float(self.labels[idx])


def collate_pair(batch):
    """
    Collate function that:
        - batches solute graphs,
        - batches solvent graphs,
        - converts temperatures and labels into tensors.
    """
    sol, solv, temps, labels = zip(*batch)
    b1 = Batch.from_data_list(list(sol))
    b2 = Batch.from_data_list(list(solv))
    temps = torch.tensor(temps, dtype=torch.float32).view(-1, 1)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    return b1, b2, temps, labels


def make_full_test_loader(csv_path: str, batch_size: int = 64) -> DataLoader:
    """
    Build a test DataLoader from a CSV, without splitting.

    Parameters
    ----------
    csv_path : str
        Path to test CSV.
    batch_size : int

    Returns
    -------
    DataLoader
        Test loader over the full dataset.
    """
    d1, d2, y, T = _smiles_graph_inputs_from_csv(csv_path)
    ds = PairDataset(d1, d2, T, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pair)


# ================== encoder loading ==================

def load_pretrained_encoders(sample, ckpt_paths: List[str]):
    """
    Load a list of pre-trained ChemSpace encoders from checkpoint paths.

    Parameters
    ----------
    sample : Data
        A sample graph used to infer node / edge feature dimensions.
    ckpt_paths : list of str
        Paths to encoder checkpoints.

    Returns
    -------
    encs : list of GNNModelWithNewLoss
        Encoder modules.
    """
    encs = []
    for path in ckpt_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")

        model = GNNModelWithNewLoss(
            num_node_features=sample.x.shape[1],
            num_edge_features=sample.edge_attr.shape[1],
            num_global_features=0,
            cov_num=3,
            hidden_dim=EMB_DIM,
        ).to(device)

        ck = torch.load(path, map_location=device)
        state = ck.get("encoder_state_dict", ck)
        model.load_state_dict(state, strict=True)
        encs.append(model)
    return encs


# ================== model definition ==================

def _stack_encode(encs: List[nn.Module], batch: Batch) -> torch.Tensor:
    """
    Encode a batch with each encoder and stack the embeddings along a new dimension.

    Parameters
    ----------
    encs : list of nn.Module
    batch : Batch

    Returns
    -------
    embs : torch.Tensor
        Shape: [batch_size, num_encoders, emb_dim]
    """
    return torch.stack([enc(batch) for enc in encs], dim=1)


class TempPairMLP(nn.Module):
    """
    MLP that combines:
        - fused solute embedding
        - fused solvent embedding
        - embedded temperature

    to predict a scalar logS value.
    """
    def __init__(
        self,
        z_dim: int,
        t_embed_dim: int = 32,
        hidden: Tuple[int, ...] = (1024, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.t_enc = nn.Sequential(nn.Linear(1, t_embed_dim), nn.ReLU())
        in_dim = z_dim * 2 + t_embed_dim

        layers: List[nn.Module] = []
        dims = [in_dim, *hidden, 1]
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.nn = nn.Sequential(*layers)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        if T.dim() == 1:
            T = T.unsqueeze(1)
        t = self.t_enc(T.to(z1.device))
        x = torch.cat([z1, z2, t], dim=-1)
        return self.nn(x).squeeze(-1)


class SolvationPredictor(nn.Module):
    """
    Full solvation regression model:

        solute encoders  -> WeightedFusion -> fused solute embedding
        solvent encoders -> WeightedFusion -> fused solvent embedding
        + temperature     -> TempPairMLP   -> scalar logS prediction
    """
    def __init__(
        self,
        sol_encs: List[nn.Module],
        solv_encs: List[nn.Module],
        sol_fusion: nn.Module,
        solv_fusion: nn.Module,
        combiner: TempPairMLP,
    ):
        super().__init__()
        self.sol_encs = nn.ModuleList(sol_encs)
        self.solv_encs = nn.ModuleList(solv_encs)
        self.sol_f = sol_fusion
        self.solv_f = solv_fusion
        self.combine = combiner

    def _fuse(self, fusion: nn.Module, embs: torch.Tensor, g: Optional[torch.Tensor]):
        out = fusion(embs, global_features=g)
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        sol_batch: Batch,
        solv_batch: Batch,
        temps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = _stack_encode(self.sol_encs, sol_batch)
        z2 = _stack_encode(self.solv_encs, solv_batch)

        g1 = getattr(sol_batch, "global_features", None)
        g2 = getattr(solv_batch, "global_features", None)

        z1 = self._fuse(self.sol_f, z1, g1)
        z2 = self._fuse(self.solv_f, z2, g2)

        pred = self.combine(z1, z2, temps)
        return pred, z1, z2


def build_model_subset(
    sample_s,
    sample_v,
    ckpt_paths: List[str],
    solute_dropout: float,
    solvent_dropout: float,
    mlp_hidden: Tuple[int, ...],
    mlp_dropout: float,
    t_embed_dim: int,
) -> SolvationPredictor:
    """
    Build a SolvationPredictor using a subset of pre-trained encoders.

    Parameters
    ----------
    sample_s, sample_v : Data
        Example solute and solvent graphs used to infer feature sizes.
    ckpt_paths : list of str
        Paths to encoder checkpoints (one per ChemSpace task).
    solute_dropout, solvent_dropout : float
        Dropout rates for solute / solvent fusion modules.
    mlp_hidden : tuple of int
        Hidden layer sizes for TempPairMLP.
    mlp_dropout : float
        Dropout rate in TempPairMLP.
    t_embed_dim : int
        Temperature embedding dimension.

    Returns
    -------
    SolvationPredictor
    """
    sol_encs = load_pretrained_encoders(sample_s, ckpt_paths)
    solv_encs = load_pretrained_encoders(sample_v, ckpt_paths)

    sol_f = WeightedFusion(len(ckpt_paths), EMB_DIM, dropout=solute_dropout, pair=True).to(device)
    solv_f = WeightedFusion(len(ckpt_paths), EMB_DIM, dropout=solvent_dropout, pair=True).to(device)

    # Probe fused embedding dimension
    with torch.no_grad():
        b1 = Batch.from_data_list([sample_s]).to(device)
        b2 = Batch.from_data_list([sample_v]).to(device)
        z1 = _stack_encode(sol_encs, b1)
        z2 = _stack_encode(solv_encs, b2)
        g1 = getattr(b1, "global_features", None)
        g2 = getattr(b2, "global_features", None)
        z1 = sol_f(z1, g1)
        z2 = solv_f(z2, g2)
        if isinstance(z1, tuple):
            z1 = z1[0]
        if isinstance(z2, tuple):
            z2 = z2[0]
        z_dim = z1.shape[-1]

    comb = TempPairMLP(
        z_dim,
        t_embed_dim=t_embed_dim,
        hidden=mlp_hidden,
        dropout=mlp_dropout,
    ).to(device)

    return SolvationPredictor(sol_encs, solv_encs, sol_f, solv_f, comb).to(device)


# ================== testing ==================

def test_loader_metrics(
    model: SolvationPredictor,
    loader: DataLoader,
) -> Tuple[float, float]:
    """
    Evaluate the model on a given loader and compute:
        - MSE
        - pct(|pred - label| <= 1)

    Returns
    -------
    mse_val : float
    pct1 : float
    """
    model.eval()
    mse = nn.MSELoss()

    preds = []
    labs = []

    with torch.no_grad():
        for s, v, t, y in loader:
            s = s.to(device)
            v = v.to(device)
            t = t.to(device)
            y = y.view(-1).to(device)

            p, _, _ = model(s, v, t)
            preds.append(p.cpu().numpy())
            labs.append(y.cpu().numpy())

    if not preds:
        return float("inf"), 0.0

    preds_np = np.concatenate(preds)
    labs_np = np.concatenate(labs)

    mse_val = float(mse(torch.tensor(preds_np), torch.tensor(labs_np)).item())
    pct1 = pct_within_1_np(preds_np, labs_np)
    return mse_val, pct1


# ================== argument parsing & main ==================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a trained pairwise solvation regression model."
    )

    # Data
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training CSV used originally (only for inferring graph dimensions).",
    )
    parser.add_argument(
        "--test_csvs",
        type=str,
        required=True,
        help=(
            "Comma-separated list of test CSV paths. "
            "Each must have solute_smiles, solvent_smiles, temperature, logS."
        ),
    )

    # Encoders & subset
    parser.add_argument(
        "--encoder_ckpts",
        type=str,
        required=True,
        help=(
            "Comma-separated list of encoder checkpoint paths. "
            "Example: '.../task_0/best_model.pth,.../task_1/best_model.pth,...'"
        ),
    )
    parser.add_argument(
        "--space_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated indices specifying which encoders to use "
            "(subset of [0..N-1]; default: all)."
        ),
    )

    # Trained pairwise model
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help=(
            "Path to trained pairwise model checkpoint. "
            "Must contain either 'model_state_dict' or a valid state_dict."
        ),
    )

    # Structural hyperparameters (must match training)
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=512,
        help="Embedding dimension of each encoder (must match training).",
    )
    parser.add_argument(
        "--solute_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for solute fusion module (must match training).",
    )
    parser.add_argument(
        "--solvent_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for solvent fusion module (must match training).",
    )
    parser.add_argument(
        "--mlp_hidden",
        type=str,
        default="1024,128,64",
        help="Comma-separated hidden layer sizes for TempPairMLP (must match training).",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.2,
        help="Dropout rate in TempPairMLP (must match training).",
    )
    parser.add_argument(
        "--t_embed_dim",
        type=int,
        default=32,
        help="Temperature embedding dimension in TempPairMLP (must match training).",
    )

    # Batch size & output
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/solvation_pair_test",
        help="Output directory for CSV summaries.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Optional experiment name used as a tag in output files.",
    )

    return parser.parse_args()


def main():
    global EMB_DIM

    args = parse_args()
    EMB_DIM = args.emb_dim

    print("[info] Using EMB_DIM =", EMB_DIM)
    ensure_dir(args.out_dir)

    # Parse encoder checkpoint list
    all_ckpts = [p.strip() for p in args.encoder_ckpts.split(",") if p.strip()]
    if not all_ckpts:
        raise ValueError("No encoder checkpoints provided via --encoder_ckpts.")

    # Parse subset indices
    if args.space_indices is None:
        subset_indices = list(range(len(all_ckpts)))
    else:
        subset_indices = [int(s.strip()) for s in args.space_indices.split(",") if s.strip()]
    if not subset_indices:
        raise ValueError("Empty subset specified via --space_indices.")

    ckpt_paths_subset = [all_ckpts[i] for i in subset_indices]
    subset_name = "_".join(str(i) for i in subset_indices)

    print(f"[info] Using encoder indices: {subset_indices}")
    print("[info] Encoder checkpoints used:")
    for idx, path in zip(subset_indices, ckpt_paths_subset):
        print(f"  - index {idx}: {path}")

    # Parse test CSVs
    test_csvs = [p.strip() for p in args.test_csvs.split(",") if p.strip()]
    if not test_csvs:
        raise ValueError("No test CSVs provided via --test_csvs.")
    test_aliases = [os.path.splitext(os.path.basename(p))[0] for p in test_csvs]

    # Use training CSV only to create a sample for inferring feature dimensions
    d1_train, d2_train, y_train, T_train = _smiles_graph_inputs_from_csv(args.train_csv)
    if len(d1_train) == 0:
        raise RuntimeError("Training CSV produced empty graph list; cannot infer dimensions.")

    sample_s = d1_train[0]
    sample_v = d2_train[0]

    # Build model structure
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
    model = build_model_subset(
        sample_s=sample_s,
        sample_v=sample_v,
        ckpt_paths=ckpt_paths_subset,
        solute_dropout=args.solute_dropout,
        solvent_dropout=args.solvent_dropout,
        mlp_hidden=mlp_hidden,
        mlp_dropout=args.mlp_dropout,
        t_embed_dim=args.t_embed_dim,
    )

    # Load trained pairwise model checkpoint
    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_ckpt}")

    ck = torch.load(args.model_ckpt, map_location=device)
    # Try "model_state_dict", fall back to assuming the ckpt itself is a state_dict
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state, strict=True)
    model.to(device)
    print(f"[info] Loaded model weights from {args.model_ckpt}")

    # Evaluate on each test CSV
    results: List[Dict[str, Any]] = []
    for csv_path, alias in zip(test_csvs, test_aliases):
        print(f"\n[info] Evaluating on test set: {csv_path} (alias='{alias}')")
        loader = make_full_test_loader(csv_path, batch_size=args.batch_size)
        mse_val, pct1 = test_loader_metrics(model, loader)

        print(
            f"[Result] Test '{alias}': "
            f"MSE={mse_val:.4f}, pct(|pred - logS| <= 1)={pct1:.3f}"
        )

        results.append(
            {
                "test_dataset": alias,
                "subset": subset_name,
                "num_encoders": len(subset_indices),
                "mse": mse_val,
                "pct_abs_err_le_1": pct1,
            }
        )

    # Save results to CSV
    base_name = os.path.splitext(os.path.basename(args.model_ckpt))[0]
    exp_tag = args.exp_name if args.exp_name else base_name
    out_path = os.path.join(
        args.out_dir,
        f"{exp_tag}_{subset_name}_test_results.csv",
    )
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n[done] Test results saved to: {out_path}")


if __name__ == "__main__":
    main()
