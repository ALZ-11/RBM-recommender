# main.py
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.reproducibility import set_seed
from src.data.pipeline import load_and_split_data
from src.models.gbrbm import GaussianBernoulliRBM
from src.training.trainer import GBRBMTrainer
from src.evaluation.evaluator import evaluate_model

# Default configuration template
DEFAULT_CONFIG = """
experiment:
  name: "GBRBM_MovieLens_Refactored"
  seed: 42
  device: "cuda"
  val_every_n_epochs: 5

dataset:
  path: "ml-latest-small/ratings.csv"
  train_split: 0.8
  test_ratio: 0.2

model:
  hidden_units: 128
  sigma: 0.1

hyperparameters:
  epochs: 30
  batch_size: 64
  learning_rate: 0.005
  cd_steps: 1
  l2_reg: 0.0002

metrics:
  top_k: 10
"""

def generate_default_config_if_missing(config_path):
    """Ensures a configuration file is present for MLOps compliance."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        print(f"Creating default config file at: {config_path}")
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG.strip())

def main(config_path="configs/config.yaml"):
    generate_default_config_if_missing(config_path)
    
    # Load settings
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Set seed for mathematical reproducibility
    set_seed(config['experiment']['seed'])
    
    device_name = config['experiment']['device']
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Executing GBRBM on device: {device}")
    
    # Execute Data Pipeline
    train_dataset, test_dataset = load_and_split_data(config)
    train_loader = DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False)
    
    # Instantiate Model and Optimizers
    model = GaussianBernoulliRBM(
        visible_units=train_dataset.num_movies,
        hidden_units=config['model']['hidden_units'],
        sigma=config['model']['sigma']
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['hyperparameters']['learning_rate']
    )
    
    trainer = GBRBMTrainer(
        model=model,
        optimizer=optimizer,
        l2_reg=config['hyperparameters']['l2_reg'],
        cd_steps=config['hyperparameters']['cd_steps']
    )
    
    # Initialize TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/{config['experiment']['name']}")
    
    # Main Model Execution Loop
    epochs = config['hyperparameters']['epochs']
    print(f"Starting CD-{config['hyperparameters']['cd_steps']} training over {epochs} epochs...\n")
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch_v, batch_m in train_loader:
            batch_v, batch_m = batch_v.to(device), batch_m.to(device)
            loss = trainer.train_step(batch_v, batch_m)
            epoch_losses.append(loss)
            
        avg_train_loss = np.mean(epoch_losses)
        writer.add_scalar("Loss/Train_Observed_MSE", avg_train_loss, epoch)
        print(f"Epoch {epoch:02d}/{epochs} | Train Reconstruction MSE: {avg_train_loss:.5f}")
        
        # Periodic evaluation on unseen target variables
        if epoch % config['experiment']['val_every_n_epochs'] == 0:
            test_results = evaluate_model(
                model=model,
                dataloader=test_loader,
                top_k=config['metrics']['top_k'],
                device=device
            )
            
            # Log metrics to TensorBoard
            writer.add_scalar("Test/RMSE", test_results['rmse'], epoch)
            writer.add_scalar("Test/Precision_at_K", test_results['precision_at_k'], epoch)
            writer.add_scalar("Test/Recall_at_K", test_results['recall_at_k'], epoch)
            writer.add_scalar("Test/NDCG_at_K", test_results['ndcg_at_k'], epoch)
            
            print(f"\n[Validation at Epoch {epoch}]")
            print(f"  Target RMSE: {test_results['rmse']:.5f}")
            print(f"  Precision@{config['metrics']['top_k']}: {test_results['precision_at_k']:.5f}")
            print(f"  Recall@{config['metrics']['top_k']}: {test_results['recall_at_k']:.5f}")
            print(f"  NDCG@{config['metrics']['top_k']}: {test_results['ndcg_at_k']:.5f}\n")
            
    writer.close()
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()