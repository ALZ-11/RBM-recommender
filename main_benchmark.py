# main_benchmark.py
import os
import yaml
import pandas as pd
from torch.utils.data import DataLoader

from src.data.pipeline import load_and_split_data
from src.evaluation.baselines import PopularityRecommender, GlobalMeanRecommender

def generate_default_config_if_missing(config_path="configs/config.yaml"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if not os.path.exists(config_path):
        from main import DEFAULT_CONFIG
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG.strip())

def run_benchmarks():
    config_path = "configs/config.yaml"
    generate_default_config_if_missing(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Execute Aligned Data Pipeline
    train_dataset, test_dataset = load_and_split_data(config)
    test_loader = DataLoader(test_dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=False)
    
    # Instantiate and Evaluate Baselines
    print("\nEvaluating Popularity Baseline...")
    pop_model = PopularityRecommender(train_dataset)
    pop_results = pop_model.evaluate(test_loader, top_k=config['metrics']['top_k'])
    
    print("Evaluating Global/Movie Mean Baseline...")
    mean_model = GlobalMeanRecommender(train_dataset)
    mean_results = mean_model.evaluate(test_loader, top_k=config['metrics']['top_k'])
    
    # Format Comparative Summary
    results = [
        {
            "Model": "Popularity Recommender",
            "Target RMSE": "N/A",
            f"Precision@{config['metrics']['top_k']}": f"{pop_results['precision_at_k']:.5f}",
            f"Recall@{config['metrics']['top_k']}": f"{pop_results['recall_at_k']:.5f}",
            f"NDCG@{config['metrics']['top_k']}": f"{pop_results['ndcg_at_k']:.5f}"
        },
        {
            "Model": "Movie Mean Recommender",
            "Target RMSE": f"{mean_results['rmse']:.5f}",
            f"Precision@{config['metrics']['top_k']}": f"{mean_results['precision_at_k']:.5f}",
            f"Recall@{config['metrics']['top_k']}": f"{mean_results['recall_at_k']:.5f}",
            f"NDCG@{config['metrics']['top_k']}": f"{mean_results['ndcg_at_k']:.5f}"
        }
    ]
    
    df = pd.DataFrame(results)
    print("\n" + "="*30 + " BENCHMARK RESULTS " + "="*30)
    print(df.to_markdown(index=False))
    print("="*79 + "\n")

if __name__ == "__main__":
    run_benchmarks()