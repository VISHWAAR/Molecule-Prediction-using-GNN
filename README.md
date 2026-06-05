# 🧪 Molecule Property Prediction using Graph Neural Networks

> Predicting **16 quantum-chemical properties** of 130,000+ molecules simultaneously — using a custom U-shaped Graph Neural Network trained on the QM9 dataset.

---

## 🔍 What This Project Does

Molecules are naturally graphs — atoms are nodes, bonds are edges. This project exploits that structure by training Graph Neural Networks (GNNs) to predict molecular properties **directly from atomic structure**, without hand-crafted descriptors.

Two progressively advanced models are built and compared:

| Model | Architecture | Edge Features | Highlights |
|---|---|---|---|
| **GINENet** | 3-layer GINE | Normalized bond distances | Baseline multi-target GNN |
| **U-GNN** | U-shaped encoder-decoder | 3D unit vectors + 16-dim RBF | Captures 3D spatial geometry |

Both models predict all **16 QM9 molecular properties** in a single forward pass — including HOMO/LUMO energies, dipole moment, polarizability, thermodynamic quantities (U, H, G), and more.

---

## 🧠 Technical Highlights

- **Dataset:** [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html) — 130,000+ small organic molecules with DFT-computed properties
- **Framework:** PyTorch + PyTorch Geometric
- **Architecture:** `GINEConv` layers with multi-layer perceptron (MLP) message functions
- **3D Geometry (U-GNN):** Edge features enriched with directional unit vectors and 16-dimensional Radial Basis Function (RBF) encodings of interatomic distances
- **Training:** Adam optimizer with `ReduceLROnPlateau` scheduler; targets normalized per-property to handle scale differences
- **Evaluation:** Per-property MAE, RMSE, and R² across all 16 targets
- **Hardware:** Google Colab T4 GPU; model checkpoint saved and reloaded for inference

---

## 🏗️ Repository Structure

```
Molecule-Prediction-using-GNN/
├── Molecule_Prediction_using_GNN.ipynb       # GINENet baseline model
└── U-GNN_A_U-Shaped_Graph_Neural_Network...  # Advanced U-GNN model
```

---

## ⚙️ Quickstart

```bash
# Install dependencies
pip install torch torch-geometric

# Run either notebook in Google Colab (T4 GPU recommended)
# Data downloads automatically via PyTorch Geometric
```

Both notebooks are self-contained — data loading, preprocessing, training, and evaluation all run end-to-end.

---

## 📊 What Gets Predicted

The models simultaneously predict all 16 QM9 targets:

| Property | Description |
|---|---|
| Dipole moment | Molecular polarity |
| Isotropic polarizability | Response to electric fields |
| HOMO / LUMO energy | Frontier molecular orbitals |
| HOMO-LUMO gap | Electronic excitation energy |
| ZPVE | Zero-point vibrational energy |
| U₀, U, H, G | Thermodynamic energies |
| Cv | Heat capacity |
| logP (implicit) | Solubility proxy |

---

## 🚀 Key Design Choices

**Why GINEConv?** Unlike vanilla GCN, GINE incorporates edge features into message passing — critical for chemistry where bond type and geometry matter.

**Why U-shaped architecture?** Inspired by U-Net in image segmentation, skip connections between encoder and decoder layers preserve fine-grained atomic features that would otherwise be lost during graph pooling.

**Why RBF distance encoding?** Raw distances are hard to learn from. Projecting them onto 16 Gaussian basis functions gives the model a smooth, rich representation of 3D space.

---

## 🛠️ Skills Demonstrated

`PyTorch` · `PyTorch Geometric` · `Graph Neural Networks` · `Molecular ML` · `Multi-task Learning` · `3D Geometry Encoding` · `Deep Learning` · `Python` · `Google Colab`

---

## 📚 References

- Ramakrishnan et al., *Scientific Data* (2014) — QM9 dataset
- Xu et al., *ICLR* (2019) — How Powerful are Graph Neural Networks? (GIN)
- Hu et al., *NeurIPS* (2020) — Strategies for Pre-training GNNs

---

## 👤 Author

**VISHWAAR**
[GitHub](https://github.com/VISHWAAR) · [Repository](https://github.com/VISHWAAR/Molecule-Prediction-using-GNN)
