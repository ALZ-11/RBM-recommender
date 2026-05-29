# RBM Movie Recommender

This project implements a collaborative recommendation engine using a **Gaussian-Bernoulli Restricted Boltzmann Machine (RBM)**.


## Methodology
The model is a Restricted Boltzmann Machine (RBM) optimized for collaborative filtering. A Gaussian-Bernoulli variant was chosen to handle continuous movie ratings (0.5 to 5.0) instead of simple binary inputs.

## Dataset
The system uses the [MovieLens small dataset](https://grouplens.org/datasets/movielens/).

## Requirements
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Ipywidgets

## Usage
1. Open `rbmRecommender.ipynb` in a Jupyter environment.
2. Run the notebook cells to:
   - Download and preprocess the dataset.
   - Train the RBM model.
   - Generate and evaluate recommendations.
   - Use the interactive search box to get top 10 recommendations for a specific User ID.
