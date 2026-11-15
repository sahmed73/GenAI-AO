"""Data loading, chemistry-group folds, descriptor scaling and metrics."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "antioxidant_descriptors.csv.gz"
TARGET = "k_pois_raw"
SIMULATION_DESCRIPTORS = (
    "diffusion_coefficient",
    "hydrogen_bond_probability",
    "buried_volume_fraction",
)

OUTER_FOLDS = 5
INNER_FOLDS = 3
TARGET_STRATA = 5
OVERSIZED_SCAFFOLD = 50
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
BUTINA_DISTANCE = 0.6
OUTER_SEED_START, OUTER_SEED_CANDIDATES = 20260804, 512
INNER_SEED_START, INNER_SEED_CANDIDATES = 20262000, 128


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(path: Path | str = DEFAULT_DATA) -> tuple[pd.DataFrame, list[str]]:
    """Return the table and the 1,990 candidate descriptors in column order."""
    data = pd.read_csv(path, float_precision="round_trip")
    rdkit = [c for c in data.columns if c.startswith("rdkit_")]
    pyl3dmd = [c for c in data.columns if c.startswith("pyl3dmd_")]
    descriptors = [*rdkit, *pyl3dmd, *SIMULATION_DESCRIPTORS]
    if len(rdkit) != 217 or len(pyl3dmd) != 1770:
        raise ValueError(f"Unexpected descriptor counts: {len(rdkit)} RDKit, {len(pyl3dmd)} PyL3dMD")
    if data["smiles"].duplicated().any() or not np.isfinite(data[TARGET]).all():
        raise ValueError("SMILES must be unique and every rate must be finite")
    return data, descriptors


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


# ---------------------------------------------------------------------------
# Chemistry groups: Bemis-Murcko scaffolds; acyclic molecules and scaffolds
# with more than 50 members are clustered by Morgan-fingerprint similarity.
# ---------------------------------------------------------------------------

def _butina(molecules, indices, prefix):
    order = sorted(indices, key=lambda i: (Chem.MolToSmiles(molecules[i]), i))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fingerprints = [generator.GetFingerprint(molecules[i]) for i in order]
    distances = []
    for i in range(1, len(fingerprints)):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend(1.0 - s for s in similarity)
    clusters = Butina.ClusterData(distances, len(fingerprints), BUTINA_DISTANCE, isDistData=True)
    return {order[m]: f"{prefix}_fp_{c:05d}" for c, members in enumerate(clusters) for m in members}


def chemistry_groups(smiles) -> np.ndarray:
    molecules = [Chem.MolFromSmiles(str(s)) for s in smiles]
    groups, acyclic = [], []
    for i, molecule in enumerate(molecules):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        groups.append(f"murcko:{scaffold}" if scaffold else None)
        if not scaffold:
            acyclic.append(i)
    for i, label in _butina(molecules, acyclic, "acyclic").items():
        groups[i] = label
    for label, count in sorted(Counter(groups).items()):
        if count <= OVERSIZED_SCAFFOLD or label.startswith("acyclic_"):
            continue
        members = [i for i, g in enumerate(groups) if g == label]
        prefix = "murcko_" + hashlib.sha1(label.encode()).hexdigest()[:10]
        for i, sub in _butina(molecules, members, prefix).items():
            groups[i] = sub
    return np.asarray(groups, dtype=str)


def _strata(target) -> np.ndarray:
    ranked = pd.Series(np.asarray(target, float)).rank(method="first")
    return np.asarray([f"q{int(v)}" for v in pd.qcut(ranked, q=TARGET_STRATA, labels=False)])


def _balance(assignment, strata, target, folds) -> float:
    ideal = len(assignment) / folds
    mean = float(np.mean(target))
    labels = sorted(set(strata))
    share = {s: float(np.mean(strata == s)) for s in labels}
    size = max(abs(np.count_nonzero(assignment == f) - ideal) / ideal for f in range(folds))
    level = max(abs(float(np.mean(target[assignment == f])) / mean - 1.0) for f in range(folds))
    stratum = max(abs(float(np.mean(strata[assignment == f] == s)) - share[s])
                  for f in range(folds) for s in labels)
    return float(size + 2.0 * stratum + level)


def _balanced_split(strata, groups, target, folds, seed_start, candidates) -> np.ndarray:
    """Among candidate seeds, keep the grouped split with the best size/target balance."""
    best = None
    dummy = np.zeros(len(strata))
    for seed in range(seed_start, seed_start + candidates):
        assignment = np.full(len(strata), -1)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, test) in enumerate(splitter.split(dummy, strata, groups)):
            assignment[test] = fold
        candidate = (_balance(assignment, strata, target, folds), seed, assignment)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2]


def assign_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Chemistry groups, five outer folds and three inner folds per outer fold."""
    groups = chemistry_groups(data["smiles"])
    target = data[TARGET].to_numpy(float)
    strata = _strata(target)
    folds = pd.DataFrame({"chemistry_group": groups}, index=data.index)
    folds["outer_fold"] = _balanced_split(strata, groups, target, OUTER_FOLDS,
                                          OUTER_SEED_START, OUTER_SEED_CANDIDATES)
    for outer in range(OUTER_FOLDS):
        train = np.flatnonzero(folds["outer_fold"].to_numpy() != outer)
        inner = np.full(len(data), -1)
        inner[train] = _balanced_split(strata[train], groups[train], target[train], INNER_FOLDS,
                                       INNER_SEED_START + 200 * outer, INNER_SEED_CANDIDATES)
        folds[f"inner_fold_{outer}"] = inner
    return folds


class FoldTransformer:
    """Median imputation, constant-feature removal, clipping and standardization,
    fitted on training molecules only."""

    def __init__(self, names, clip_sigma: float = 10.0):
        self.names = list(names)
        self.clip_sigma = clip_sigma

    def fit(self, frame: pd.DataFrame) -> "FoldTransformer":
        values = frame[self.names].to_numpy(float)
        values[~np.isfinite(values)] = np.nan
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        finite = np.isfinite(medians)
        self.medians_ = np.where(finite, medians, 0.0)
        values = np.where(np.isfinite(values), values, self.medians_)
        sd = values.std(axis=0)
        self.keep_ = finite & np.isfinite(sd) & (sd > 1e-12)
        kept = values[:, self.keep_]
        mean, sd = kept.mean(axis=0), kept.std(axis=0)
        self.lower_ = np.maximum(kept.min(axis=0), mean - self.clip_sigma * sd)
        self.upper_ = np.minimum(kept.max(axis=0), mean + self.clip_sigma * sd)
        kept = np.clip(kept, self.lower_, self.upper_)
        self.mean_, self.sd_ = kept.mean(axis=0), kept.std(axis=0)
        self.output_names_ = np.asarray(self.names, dtype=object)[self.keep_]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.names].to_numpy(float)
        values = np.where(np.isfinite(values), values, self.medians_)
        kept = np.clip(values[:, self.keep_], self.lower_, self.upper_)
        return ((kept - self.mean_) / self.sd_).astype(np.float32)


def group_bootstrap_interval(values, groups, statistic, replicates=2000, seed=0):
    """95% interval from resampling chemistry groups with replacement."""
    values = np.asarray(values)
    labels, index = np.unique(np.asarray(groups, str), return_inverse=True)
    members = [np.flatnonzero(index == g) for g in range(len(labels))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        chosen = np.concatenate([members[g] for g in rng.integers(0, len(labels), len(labels))])
        draws.append(statistic(chosen))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
