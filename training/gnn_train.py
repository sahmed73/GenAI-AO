#!/usr/bin/env python3
"""Graph neural network fit and substructure masking.

Three independently initialized GINE models are trained on all molecules for a
fixed number of epochs. For substructure masking, each model is applied to the
complete molecule; the atoms of a chemical group (SMARTS) are then left out
when the atom representations are pooled into the prediction. The attribution
of a group is the full-molecule prediction minus the masked prediction,
averaged over the three models and over the occurrences of the group within a
molecule.

Run from the repository root:  python training/gnn_train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from common import DEFAULT_DATA, TARGET, chemistry_groups, load_dataset, metrics, seed_everything
from graph import smiles_to_graph
from model import GINE

SEEDS = (98001, 98002, 98003)
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, CLIP_NORM = 64, 5e-4, 5e-4, 5.0
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 2000, 20260925

# (name, SMARTS, positions in the match that are masked)
GROUPS = [
    ("Phenolic OH", "[OX2H]-[c]", (0,)),
    ("tert-Butyl", "[*]-[C]([CH3])([CH3])[CH3]", (1, 2, 3, 4)),
    ("Methoxy", "[c]-[O]-[CH3]", (1, 2)),
    ("Aryl ether", "[c]-[O]-[#6;!CH3]", (1,)),
    ("Carboxylic acid", "[C](=[O])[OH]", (0, 1, 2)),
    ("Ester", "[#6,c]-[C](=[O])-[O]-[#6]", (1, 2, 3)),
    ("Amide", "[C](=[O])-[N;X3]", (0, 1, 2)),
    ("Amine", "[N;X3;!$(N-C=O);!$(N-C=S)]", (0,)),
    ("Ketone", "[#6,c]-[C](=[O])-[#6,c]", (1, 2)),
    ("Aldehyde", "[#6,c]-[CH]=[O]", (1, 2)),
    ("Nitrile", "[C]#[N]", (0, 1)),
    ("Alkene", "[C]=[C]", (0, 1)),
    ("Halogen", "[F,Cl,Br,I]", (0,)),
    ("Sulfur", "[S,s]", (0,)),
]
NOT_RANKED = {"Phenolic OH"}  # present in every molecule


def group_occurrences(smiles: str):
    """[(group, occurrence, masked atom indices)] for one molecule."""
    molecule = Chem.MolFromSmiles(smiles)
    found = []
    for name, smarts, positions in GROUPS:
        query = Chem.MolFromSmarts(smarts)
        for j, match in enumerate(molecule.GetSubstructMatches(query, uniquify=True)):
            atoms = tuple(sorted({match[p] for p in positions}))
            if len(atoms) < molecule.GetNumAtoms():
                found.append((name, j, atoms))
    return found


def train_models(graphs, y, epochs, snapshots, device):
    """Train one model per seed; return {epoch: [state_dict per seed]}."""
    mean, sd = float(y.mean()), max(float(y.std()), 1e-8)
    states = {e: [] for e in snapshots}
    for seed in SEEDS:
        seed_everything(seed)
        examples = []
        for graph, value in zip(graphs, y):
            example = graph.clone()
            example.y = torch.tensor([(value - mean) / sd], dtype=torch.float32)
            examples.append(example)
        loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        model = GINE(graphs[0].x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for epoch in range(1, max(snapshots) + 1):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(batch), batch.y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                optimizer.step()
            if epoch in states:
                states[epoch].append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if epoch % 10 == 0:
                print(f"seed {seed}: epoch {epoch}", flush=True)
    return states, mean, sd


@torch.no_grad()
def masked_predictions(state_dicts, graphs, variants, mean, sd, device, chunk=32):
    """Mean prediction over models for every (molecule, mask) variant."""
    total = None
    for state in state_dicts:
        model = GINE(graphs[0].x.shape[1]).to(device)
        model.load_state_dict(state)
        model.eval()
        values = []
        for start in range(0, len(graphs), chunk):
            batch = Batch.from_data_list(graphs[start:start + chunk]).to(device)
            atom_states = model.atom_states(batch)
            for local in range(batch.num_graphs):
                left, right = int(batch.ptr[local]), int(batch.ptr[local + 1])
                masks = torch.ones(len(variants[start + local]), right - left, device=device)
                for j, atoms in enumerate(variants[start + local]):
                    masks[j, list(atoms)] = 0.0
                prediction = model.masked_readout([s[left:right] for s in atom_states], masks)
                values.append(prediction.cpu().numpy() * sd + mean)
        total = values if total is None else [a + b for a, b in zip(total, values)]
    return [v / len(state_dicts) for v in total]


def attributions(state_dicts, data, graphs, occurrences, mean, sd, device):
    variants = [[()] + [atoms for _, _, atoms in found] for found in occurrences]
    predictions = masked_predictions(state_dicts, graphs, variants, mean, sd, device)
    full = np.array([p[0] for p in predictions])
    rows = [{"smiles": s, "group": name, "occurrence": j, "attribution": p[0] - p[i + 1]}
            for s, found, p in zip(data.smiles, occurrences, predictions)
            for i, (name, j, _) in enumerate(found)]
    per_molecule = (pd.DataFrame(rows).groupby(["smiles", "group"], as_index=False)
                    .agg(occurrences=("occurrence", "size"), attribution=("attribution", "mean")))
    return full, per_molecule


def summarize(per_molecule, chemistry):
    per_molecule = per_molecule.assign(chemistry_group=per_molecule.smiles.map(chemistry))
    means = per_molecule.groupby("group").attribution.mean().sort_values()
    order = [g for g in means.index if g not in NOT_RANKED] + [g for g in means.index if g in NOT_RANKED]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for name in order:
        one = per_molecule[per_molecule.group == name]
        labels, index = np.unique(one.chemistry_group.to_numpy(str), return_inverse=True)
        sums = np.bincount(index, weights=one.attribution.to_numpy(float), minlength=len(labels))
        counts = np.bincount(index, minlength=len(labels))
        draw = rng.integers(len(labels), size=(BOOTSTRAP_REPLICATES, len(labels)))
        boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
        rows.append({"group": name, "molecules": len(one), "chemistry_groups": len(labels),
                     "mean_attribution": one.attribution.mean(), "median_attribution": one.attribution.median(),
                     "fraction_negative": (one.attribution < 0).mean(),
                     "ci95_low": np.quantile(boot, 0.025), "ci95_high": np.quantile(boot, 0.975)})
    summary = pd.DataFrame(rows)
    ranked = summary[~summary.group.isin(NOT_RANKED)]
    summary["rank"] = summary.group.map(dict(zip(ranked.group, ranked.mean_attribution.rank(method="first").astype(int))))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/gnn"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--sweep", type=int, nargs="*", default=[],
                        help="additional training lengths at which fit and group ranking are reported")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data)
    y = data[TARGET].to_numpy(float)
    graphs = [smiles_to_graph(s) for s in data.smiles]
    occurrences = [group_occurrences(s) for s in data.smiles]
    chemistry = dict(zip(data.smiles, chemistry_groups(data.smiles)))
    snapshots = sorted(set(args.sweep) | {args.epochs})
    print(f"{len(data)} molecules; training {len(SEEDS)} models to epoch {max(snapshots)} on {device}", flush=True)
    states, mean, sd = train_models(graphs, y, args.epochs, snapshots, device)

    args.output.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for epoch in snapshots:
        full, per_molecule = attributions(states[epoch], data, graphs, occurrences, mean, sd, device)
        summary = summarize(per_molecule, chemistry)
        fit = metrics(y, full)
        sweep_rows += [{"epochs": epoch, "training_r2": fit["r2"], **row}
                       for row in summary[["group", "mean_attribution", "rank"]].to_dict("records")]
        if epoch == args.epochs:
            pd.DataFrame({"smiles": data.smiles, "k_simulated": y, "k_fitted": full}).to_csv(
                args.output / "fitted_rates.csv", index=False)
            per_molecule.to_csv(args.output / "molecule_attributions.csv", index=False)
            summary.to_csv(args.output / "group_attributions.csv", index=False)
            (args.output / "training_fit.json").write_text(json.dumps(
                {"epochs": epoch, "molecules": len(data), **{f"training_{k}": v for k, v in fit.items()}}, indent=2) + "\n")
            print(summary.to_string(index=False), flush=True)
    if args.sweep:
        pd.DataFrame(sweep_rows).to_csv(args.output / "epoch_sweep.csv", index=False)


if __name__ == "__main__":
    main()
