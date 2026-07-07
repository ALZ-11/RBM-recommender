# Masked Gaussian-Bernoulli RBM for Collaborative Filtering

A modular PyTorch implementation of a **Masked Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM)** optimized for collaborative filtering on the MovieLens dataset. 

In order to address the severe sparsity inherent in collaborative filtering, this implementation introduces a masked Contrastive Divergence ($CD-k$) training loop, an active-item decoupled weight decay mechanism, and a memory-efficient sparse data pipeline, alongside a leakage-free evaluation protocol.

---

## Technical Features

* **Masked Contrastive Divergence ($CD-k$):** Restricts positive and negative Gibbs sampling updates exclusively to observed ratings using binary user-item masks. Unobserved items do not contribute to visible-to-hidden activations or reconstruction gradients.
* **Active-Item Decoupled Weight Decay:** Implements an AdamW-style weight decay step that is applied *only* to items actively present in the current mini-batch, preventing the latent features of dormant items from decaying prematurely to zero.
* **Exact Gradient Scaling:** Preserves the formal $1/\sigma$ scaling for weight updates and the $1/\sigma^2$ scaling for visible bias updates, ensuring balanced gradient dynamics across different noise thresholds.
* **Bayesian Shrinkage Initialization:** Initializes the model's visible biases using item averages smoothed with a Bayesian shrinkage prior ($k=5.0$) computed over the training user cohort.
* **Leakage-Free Evaluation Split:** Partitions users strictly into disjoint train and test splits. For evaluation, test users are divided into an input visible history (context) and a held-out target set (ground truth) using a deterministic, seed-based user-level random number generator.
* **Memory-Efficient Data Pipeline:** Stores sparse ratings in memory as a list of dictionaries rather than allocating dense $N \times M$ matrices, scaling well to sparse item spaces.
* **Experiment Management:** Features TensorBoard logging, YAML-based configuration, fully reproducible seeding, and baseline benchmarking.

---

## Repository Structure

```text
.
├── configs/
│   └── config.yaml                 # Experiment and hyperparameter configurations
├── src/
│   ├── data/
│   │   ├── dataset.py              # Sparse representation and context/target partition logic
│   │   └── pipeline.py             # User splitting and cold-start item filtering
│   ├── evaluation/
│   │   ├── baselines.py            # Popularity and Movie Mean baselines with Bayesian shrinkage
│   │   ├── evaluator.py            # Forward pass context-evaluation loop
│   │   └── metrics.py              # Scaled RMSE, Precision@K, Recall@K, and NDCG@K
│   ├── models/
│   │   └── gbrbm.py                # Gaussian-Bernoulli RBM architecture
│   ├── training/
│   │   └── trainer.py              # Manual gradient CD-k and active-item weight decay
│   └── utils/
│       └── reproducibility.py      # Seed configuration across Python, NumPy, and PyTorch
├── main.py                         # Training script for GBRBM
├── main_benchmark.py               # Evaluation script for baseline models
├── requirements.txt                # Dependency list
└── README.md                       # Documentation
```

---

## Model & Mathematical Overview

The core recommendation engine relies on a modified Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM), where continuous visible units represent normalized movie ratings, and binary hidden units model latent user preferences.

The energy function of the masked Gaussian-Bernoulli RBM for a given visible rating vector $\mathbf{v}$ and binary hidden vector $\mathbf{h}$ is defined as:

$$E(\mathbf{v}, \mathbf{h}) = \sum_{i} \frac{(v_i - a_i)^2}{2\sigma^2} m_i - \sum_{j} b_j h_j - \sum_{i,j} \frac{v_i}{\sigma} W_{ji} h_j m_i$$

where $m_i \in \{0, 1\}$ is the binary observation mask for item $i$, $a_i$ is the visible bias, $b_j$ is the hidden bias, and $\sigma$ is the homoscedastic visible standard deviation. 

### Conditional Expectations
1. **Hidden Units:**
   $$P(h_j = 1 \mid \mathbf{v}) = \sigma_s \left( b_j + \sum_{i} W_{ji} \frac{v_i m_i}{\sigma} \right)$$
2. **Visible Units:**
   $$E[v_i \mid \mathbf{h}] = a_i + \sigma \sum_{j} W_{ji} h_j$$

where $\sigma_s(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid function.

### Optimization & Active-Mask Decay
The parameters are updated using manual gradient computation following the Contrastive Divergence approximation, normalized by the batch size and scaled by the noise parameter $\sigma$. 

Post-update, the weights associated with active items in the batch are decayed:

$$\mathbf{W}_{*, i} \leftarrow \mathbf{W}_{*, i} \left(1 - \eta \cdot \lambda \cdot \mathbb{I}(\text{item } i \text{ active})\right)$$

where $\eta$ is the learning rate, $\lambda$ is the L2 decay coefficient, and $\mathbb{I}$ is the batch activity indicator.

---

## Dataset

This project is configured for the **MovieLens Latest Small** dataset.

Dataset statistics:
* **610 users**
* **9,742 movies**
* **100,836 ratings**

Ratings are normalized from their raw scale $[0.5, 5.0]$ to the continuous range $[0.0, 1.0]$ for model training and un-normalized back during evaluation. The data pipeline will automatically search for and unpack standard `ml-latest-small.zip` archives if present in the root folder.

---

## Evaluation Protocol & Metrics

The pipeline implements a within-user holdout evaluation protocol to simulate realistic recommendation scenarios:

1. **User Partitioning:** Users are split into disjoint training (80%) and testing (20%) cohorts.
2. **Within-User Split:** For test users, a fraction of their ratings (e.g., 20%) is masked and held out as the prediction target (ground truth).
3. **Inference (Context Phase):** The remaining 80% of the test user's ratings serve as the observable history (context) to generate latent hidden activations.
4. **Ranking & Evaluation:** The model reconstructs ratings for all unobserved items. Previously observed context items are excluded from ranking, and recommendations are evaluated against the held-out targets.

### Evaluation Metrics
Performance is measured using both rating prediction and ranking metrics:
* **Root Mean Squared Error (RMSE):** Calculated strictly over target items.
* **Precision@K & Recall@K:** Measures the relevance of the top $K$ recommendations.
* **NDCG@K (Normalized Discounted Cumulative Gain):** Evaluates ranking quality.

---

## Baseline Models

The repository includes two baseline recommenders to contextualize GBRBM performance:
* **Popularity Recommender:** Recommends the overall most popular items.
* **Movie Mean Recommender:** Predicts rating values using item averages smoothed with a Bayesian shrinkage prior to handle sparse ratings.

Both baselines are computed exclusively from the training user split to prevent information leakage.

---

## Installation

Ensure you have Python 3.10+ installed, then clone the repository and install dependencies:

```bash
git clone https://github.com/ALZ-11/RBM-recommender
cd RBM-recommender
pip install -r requirements.txt
```

### Dataset Setup
Download the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/) and place the `ml-latest-small.zip` file or the unpacked `ml-latest-small/` directory in the repository root.

---

## Usage

### 1. Training the GBRBM
Run the training pipeline:

```bash
python main.py
```
This script will load parameters from `configs/config.yaml`, partition the dataset, train the model using masked $CD-k$ updates, and run evaluations on held-out targets at configured intervals.

### 2. Evaluating Benchmarks
Run the baseline evaluation suite:

```bash
python main_benchmark.py
```
This script computes comparative evaluation metrics for the Popularity and Movie Mean baselines using identical dataset partitions and seeds.

---

## Configuration

All model architecture parameters, optimization hyperparameters, and evaluation protocols are configured via a centralized YAML file (`configs/config.yaml`).

Example structure:

```yaml
experiment:
  name: "GBRBM_MovieLens_Tuned"
  seed: 42
  device: "cuda"              # 'cuda' or 'cpu'
  val_every_n_epochs: 5       # Run evaluation loop on held-out targets every N epochs

dataset:
  path: "ml-latest-small/ratings.csv"
  train_split: 0.8            # User-level split ratio
  test_ratio: 0.2             # Within-user rating hold-out ratio (gamma)

model:
  hidden_units: 256           # Number of latent features
  sigma: 0.05                 # Standard deviation of Gaussian visible units

hyperparameters:
  epochs: 40
  batch_size: 64
  learning_rate: 0.005
  cd_steps: 3                 # Contrastive Divergence steps (k)
  l2_reg: 0.001               # Weight decay coefficient

metrics:
  top_k: 10                   # Top K recommendations to evaluate
```

---

## TensorBoard Logging

Training progress and evaluation performance are tracked automatically. Run the following command to view training curves:

```bash
tensorboard --logdir runs
```

Monitored parameters include training reconstruction loss, RMSE, Precision@K, Recall@K, and NDCG@K.

---

## Reproducibility

In order to ensure deterministic experiments, the environment configuration initializes and fixes:
* Python random seed
* NumPy random seed
* PyTorch random seed (CPU & CUDA)
* CuDNN deterministic execution behaviors

---

## Future Work

Potential areas for extending this codebase:
* Integrating a Matrix Factorization (SVD) baseline
* Implementing Neural Collaborative Filtering (NeuMF) and LightGCN comparison models
* Automating hyperparameter tuning using Optuna
* Evaluating scalability on larger datasets like MovieLens-1M and MovieLens-20M
* Adding statistical significance testing (e.g., paired t-tests) on metric improvements
* Implementing additional ranking metrics such as MAP (Mean Average Precision) and MRR (Mean Reciprocal Rank)