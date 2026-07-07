# Masked Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM) for Collaborative Filtering

This repository implements a collaborative recommendation engine utilizing a **Masked Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM)** in PyTorch. 

Unlike some traditional energy-based recommender approaches that treat unobserved movie ratings as explicit $0.0$ scores (which heavily biases the network's parameters toward low ratings in highly sparse environments), this implementation introduces a parallel **dual-tensor masking framework**. Unobserved interactions contribute exactly zero gradient pressure during training, allowing the model to learn strictly from observed preference manifolds.

---

## 1. Theoretical & Mathematical Foundations

### The Masked Energy Function
For any user, let $\mathbf{v} \in \mathbb{R}^D$ be the continuous normalized ratings vector across $D$ movies, $\mathbf{h} \in \{0, 1\}^F$ be the binary latent features, and $\mathbf{m} \in \{0, 1\}^D$ be the user's observed rating mask. The system's energy is defined as:

$$E(\mathbf{v}, \mathbf{h} \mid \mathbf{m}) = \sum_{i=1}^D m_i \frac{(v_i - b_i)^2}{2\sigma^2} - \sum_{j=1}^F c_j h_j - \sum_{i=1}^D \sum_{j=1}^F m_i \frac{v_i}{\sigma} W_{ij} h_j$$

where:
*   $W \in \mathbb{R}^{F \times D}$ is the latent weight matrix.
*   $\mathbf{b} \in \mathbb{R}^D$ and $\mathbf{c} \in \mathbb{R}^F$ are the visible and hidden bias vectors, respectively.
*   $\sigma \in \mathbb{R}^+$ is the standard deviation scaling factor for continuous visibles.

### Gibbs Sampling Conditionals
Because the energy function has no quadratic cross-products coupling hidden units ($h_j h_k$ for $j \neq k$), the conditional probability distribution $P(\mathbf{h} \mid \mathbf{v}; \mathbf{m})$ factorizes. This yields standard independent sigmoid activations:

$$P(h_j = 1 \mid \mathbf{v} ; \mathbf{m}) = \sigma_L\left( c_j + \sum_{i=1}^D m_i \frac{v_i}{\sigma} W_{ij} \right)$$

For the visible layer, the conditional probability for any observed rating ($m_i = 1$) is a Gaussian distribution:

$$P(v_i \mid \mathbf{h} ; \mathbf{m}) \sim \mathcal{N}\left( b_i + \sigma \sum_{j=1}^F W_{ij} h_j, \sigma^2 \right)$$

Unobserved visible entries ($m_i = 0$) are omitted during sampling, protecting the parameter updates from arbitrary imputation bias.

---

## 2. Data Engineering & Leakage Prevention

### Dual-Tensor Global Normalization
Raw movie ratings are scaled globally from their raw interval $[0.5, 5.0]$ to the continuous $[0.0, 1.0]$ interval:
$$v_{u, i} = \frac{r_{u, i} - 0.5}{4.5}$$
Unrated items are initialized to $0.0$ in the visible vector, but their parallel mask value is set to $m_{u, i} = 0.0$.

### Leak-Free Splitting
In order to ensure that performance metrics represent true generalization to unseen data, we implement a partitioning protocol:
1.  **User Split:** Users are partitioned into $80\%$ training users and $20\%$ test users.
2.  **Within-User Test Split:** For each test user $u$, their rated indices are randomly partitioned into:
    *   **Input History ($80\%$ of indices):** Handled as the visible context.
    *   **Held-out Targets ($20\%$ of indices):** Hidden during Gibbs sampling and used strictly to calculate continuous RMSE and ranking metrics (Recall@K, NDCG@K).
3.  **Recommendation Filtering:** When calculating ranking metrics, movies present in the user's Input History are masked out, preventing the model from recommending already-watched movies.

---

## 3. Repository Architecture

```text
RBM-Recommender/
├── configs/
│   └── config.yaml             # External hyperparameter configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py          # Dual-tensor Dataset class and splitting logic
│   │   └── pipeline.py         # Data download, mapping, and split wrapper
│   ├── models/
│   │   └── gbrbm.py            # Custom GBRBM PyTorch Module
│   ├── training/
│   │   └── trainer.py          # CD-k Trainer with custom masked gradients
│   └── evaluation/
│       ├── metrics.py          # Continuous and ranking metrics (RMSE, NDCG)
│       └── evaluator.py        # Evaluation loop over test loaders
├── main.py                     # Main entry point and orchestration runner
├── requirements.txt
└── README.md
```

---

## 4. Execution & Reproducibility Guide

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/ALZ-11/RBM-Recommender
cd RBM-Recommender
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the MovieLens Small dataset (`ml-latest-small.zip`) from GroupLens and place the archive directly in the root directory. The pipeline will automatically unpack the contents upon execution.

### 3. Run Training and Evaluation
Run the central script to start model training under the default configuration:
```bash
python main.py
```
This automatically writes a default YAML config file at `configs/config.yaml` if missing, initializes the database, and begins training with periodic metric reporting on TensorBoard.

### 4. Visualizing Results
Execute the following command to monitor loss curves and continuous/ranking metrics on TensorBoard:
```bash
tensorboard --logdir=runs/
```