# PHI-HCL: Protein-Metabolite Interaction via HyperCL

PHI-HCL is a comprehensive toolkit and dataset collection for studying protein-metabolite interactions using hypergraph-based deep learning. It is designed for researchers in bioinformatics, computational biology, and drug discovery.

## Model Architecture Overview
### The proposed framework consists of three main stages: (A) Raw Feature Extraction, (B) Feature Improvement, and (C) Prediction.
![Model Architecture](https://github.com/ztjin958/PMI-HCL/blob/main/Figure%201_01.png)
#### (A) Raw Feature Extraction
In the initial stage, raw biological data is transformed into high-dimensional feature representations:
- Metabolites: SMILES strings are processed using ChemGPT to generate the initial metabolite feature matrix ($FM$).
- Proteins: Amino acid sequences are encoded using ProtT5 to generate the initial protein feature matrix ($FP$).
#### (B) Feature Improvement
This stage enhances the raw features by integrating relational knowledge and fine-tuning representations through multi-scale analysis:
- Graph-based Knowledge Integration:
	* Structural information from STITCH (for metabolites) and STRING (for proteins) is utilized.K-Nearest Neighbors (KNN) is applied to construct hypergraphs ($HG_m$ and $HG_p$) at different scales($K_1, K_2$).
- Hypergraph Convolution & Contrastive Learning:
	* Dual HyperConv layers extract complex higher-order correlations.Contrastive Learning is employed between different scales to ensure robust and invariant feature learning.
- Channel-wise Attention Mechanism:
	* Features from different scales ($\widetilde{FM}$, $\widetilde{FP}$) undergo Row Average Pooling followed by a Fully Connected Neural Network.
	* The model dynamically re-weights feature channels to highlight the most informative biological signals, resulting in refined features($\ddot{FM}$ and $\ddot{FP}$).
	* Finally, 1-D Convolutional Neural Networks (1-D CNN) are used to reduce dimensionality and consolidate the enhanced features ($\widehat{FM}$ and $\widehat{FP}$).
#### (C) Prediction
The final stage performs the interaction inference:
- Feature Fusion: The raw features ($FM, FP$) and the improved features ($\widehat{FM}, \widehat{FP}$) are integrated to form the final comprehensive representations.
- Interaction Scoring: The fused features for both metabolites and proteins are fed into a Fully Connected Layer (Multi-layer Perceptron) to predict the probability of interaction.

## Background

Understanding protein-metabolite interactions is crucial for elucidating biological processes and drug mechanisms. This project provides curated datasets and PyTorch-based code for building, training, and evaluating hypergraph neural network models on multiple species.

## Features

- Ready-to-use datasets for human, E. coli, yeast
- Hypergraph neural network (HGNN) model implementation (PyTorch & torch-geometric)
- Data preprocessing and feature extraction scripts
- Reproducible training and evaluation pipeline
- Large file support (split for GitHub compatibility)

## Directory Structure

```
piazza/           # E. coli dataset
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
- **meta.smi**: metabolite SMILES strings
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



