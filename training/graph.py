"""Molecular graphs from SMILES: atoms are nodes and bonds are edges."""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

ELEMENTS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
HYDROGENS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, allowed) -> list[float]:
    """One-hot encoding with a final 'other' position."""
    encoded = [0.0] * (len(allowed) + 1)
    encoded[allowed.index(value) if value in allowed else -1] = 1.0
    return encoded


def atom_features(atom: Chem.Atom) -> list[float]:
    phenolic_oxygen = atom.GetAtomicNum() == 8 and any(n.GetIsAromatic() for n in atom.GetNeighbors())
    return [
        *one_hot(atom.GetAtomicNum(), ELEMENTS),
        *one_hot(atom.GetDegree(), DEGREES),
        *one_hot(atom.GetTotalNumHs(), HYDROGENS),
        *one_hot(atom.GetHybridization(), HYBRIDIZATIONS),
        atom.GetFormalCharge() / 2.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 200.0,
        float(atom.GetNumRadicalElectrons()),
        float(phenolic_oxygen),
    ]


def smiles_to_graph(smiles: str) -> Data:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    x = torch.tensor([atom_features(a) for a in molecule.GetAtoms()], dtype=torch.float32)
    edges, types = [], []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        kind = BONDS.index(bond.GetBondType()) if bond.GetBondType() in BONDS else len(BONDS)
        edges += [[i, j], [j, i]]
        types += [kind, kind]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=torch.tensor(types, dtype=torch.long))
