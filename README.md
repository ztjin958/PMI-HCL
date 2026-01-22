# PHI-HCL: Protein-Metabolite Interaction via HyperCL

PHI-HCL is a comprehensive toolkit and dataset collection for studying protein-metabolite (compound) interactions using hypergraph-based deep learning. It is designed for researchers in bioinformatics, computational biology, and drug discovery.

## Background

Understanding protein-metabolite interactions is crucial for elucidating biological processes and drug mechanisms. This project provides curated datasets and PyTorch-based code for building, training, and evaluating hypergraph neural network models on multiple species.

## Features

- Ready-to-use datasets for human, E. coli, yeast, and more
- Hypergraph neural network (HGNN) model implementation (PyTorch & torch-geometric)
- Data preprocessing and feature extraction scripts
- Reproducible training and evaluation pipeline
- Large file support (split for GitHub compatibility)

## Directory Structure

```
piazza/           # Protein-related dataset (FASTA, links, features, etc.)
PMIDB/human/      # Human protein-metabolite interaction data
stitch_ecoli/     # E. coli dataset
stitch_yeast/     # Yeast dataset
Model.py          # Model definition (HGNN)
Prepare.py        # Data preprocessing and feature generation
main.py           # Main training & evaluation script
utils.py          # Utility functions (metrics, matrix conversion, etc.)
requirements.txt  # Python dependencies
```

## Dataset Details

- **edges.csv / m_p_links.csv**: Protein-metabolite interaction pairs
- **m_m_links.tsv / p_p_links.tsv**: Metabolite-metabolite & protein-protein similarity networks
- **meta.smi**: Compound SMILES strings
- **meta_ChemGPT-19M.npy / protein_large_model.npy**: Precomputed feature matrices
- **protein_edge.edgelist / meta_edge.edgelist**: Graph edge lists for network construction
- **p_p_links_part*.tsv**: Large files split for GitHub upload

## Model & Code Overview

- `Model.py`: Implements a hypergraph neural network (HGNN) for learning on protein/metabolite graphs
- `Prepare.py`: Loads and processes raw data, builds graph structures, generates features
- `main.py`: Orchestrates training, cross-validation, and evaluation; supports GPU/CPU
- `utils.py`: Helper functions for matrix conversion, metrics, and data handling

## Installation & Usage

1. Clone the repository:
	```bash
	git clone https://github.com/ztjin958/PMI-HCL.git
	cd PMI-HCL
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Prepare data (if needed, see Prepare.py for details)
4. Run the main script:
	```bash
	python main.py
	```

## Requirements

- Python 3.7+
- numpy, pandas, tqdm, torch, torch-geometric, scikit-learn

## Citation & Contribution

If you use this project, please cite:
https://github.com/ztjin958/PMI-HCL

Contributions, issues, and pull requests are welcome!
