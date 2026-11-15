# A Generative AI and Reactive Molecular Dynamics Screening Framework for Phenolic Lubricant Antioxidants

This repository contains data and a trained molecular generation model associated with the manuscript. The study combines molecular generation, screening, reactive molecular dynamics (MD), and machine learning to investigate phenolic lubricant antioxidants.

## Manuscript

**A Generative AI and Reactive Molecular Dynamics Screening Framework for Phenolic Lubricant Antioxidants**

Shihab Ahmed, Stefan J. Eder, Mohamed Musthafa Iqbal, Nicole Dörr, and Ashlie Martini

## Repository contents

- `data/reference_antioxidants.csv`: structures and classifications of 93 reference antioxidants.
- `data/antioxidant_descriptors.csv.gz`: SMILES, simulated reverse reaction rates, and molecular descriptors for 3,000 candidates.
- `models/phenolic_digress.ckpt`: trained DiGress molecular generation model.
- `md_examples/`: three example MD systems, with reaction inputs and a trajectory for each molecule.

Extract the descriptor dataset with:

```bash
gzip -dk data/antioxidant_descriptors.csv.gz
```

## Molecular dynamics simulations

The example simulations use LAMMPS with the REACTER protocol and the PCFF force field. Each example directory contains an input script, an equilibrated configuration, reaction templates, a reaction map, and a compressed trajectory.

With a compatible LAMMPS build, the reactive input can be launched from an example directory using:

```bash
lmp -var resume 0 -var rfile none -in run.in
```

## Molecular generation

The checkpoint contains the trained DiGress model used for molecular generation. A compatible DiGress generation workflow can load it using `--checkpoint models/phenolic_digress.ckpt`.

## ANN and GNN training

The training scripts use the descriptor dataset to train an ANN and a graph neural network (GNN) for reaction-rate prediction. The GNN uses SMILES to construct molecular graphs.

Create a Python environment and install the dependencies:

```bash
conda create -n genai-ao -c conda-forge python=3.10 rdkit numpy pandas scikit-learn pytorch
conda activate genai-ao
pip install torch-geometric shap
```

Run the scripts from the repository root:

```bash
python training/ann_ranking_comparison.py
python training/ann_train.py
python training/gnn_train.py
```

Use `--data` to specify a different input CSV.
