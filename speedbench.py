# faiss
import faiss
# sys
import time
import math
from typing import Optional
import os

# usual suspects
import torch
import typer
import numpy as np

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# fast-pytorch-kmeans
import importlib.util
fast_pytorch_kmeans_available = importlib.util.find_spec("fast_pytorch_kmeans") is not None
if fast_pytorch_kmeans_available and torch.cuda.is_available():
    from fast_pytorch_kmeans import KMeans
else:
    print("No GPU available or fast_pytorch_kmeans not installed, skipping fast-pytorch-kmeans...")

# ~us~
from fastkmeans import FastKMeans

def generate_synthetic_data(n_samples, n_clusters, n_features=128, seed=42, random_clusters=False):
    """
    Generate synthetic clustering data.
    
    Args:
        n_samples: Number of data points to generate
        n_clusters: Number of clusters
        n_features: Number of features per data point
        seed: Random seed for reproducibility
        random_clusters: If True, generate completely random data without cluster structure.
                         If False, generate data centered around cluster centroids.
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features, {n_clusters} clusters...")
    
    np.random.seed(seed)
    
    if random_clusters:
        # Generate completely random data without cluster structure
        X = np.random.randn(n_samples, n_features).astype(np.float32) * 10
        # Assign random cluster labels
        cluster_indices = np.random.randint(0, n_clusters, size=n_samples)
    else:
        # Generate data centered around cluster centroids
        centers = np.random.randn(n_clusters, n_features).astype(np.float32) * 10
        X = np.empty((n_samples, n_features), dtype=np.float32)
        batch_size = 100000  
        cluster_indices = np.random.randint(0, n_clusters, size=n_samples)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_size_actual = end_idx - i
            batch_centers = centers[cluster_indices[i:end_idx]]
            noise = np.random.randn(batch_size_actual, n_features).astype(np.float32) * 5.0
            X[i:end_idx] = batch_centers + noise
    
    print(f"Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features, {n_clusters} clusters")
    return X.astype(np.float32), cluster_indices

def run_fastkmeans(data, k, max_iters=20, seed=42, max_points_per_centroid=1_000_000_000, verbose=False, device='cpu'):
    """Run our PyTorch KMeans implementation."""
    print(f"\n=== PyTorch KMeans on {data.shape[0]} samples, {k} clusters ===")
    n_features = data.shape[1]
    
    # Create and train the model
    import math
    kmeans = FastKMeans(
        d=n_features,
        k=k,
        niter=max_iters,
        tol=-math.inf,
        device=device,
        gpu=torch.cuda.is_available() and device != 'cpu',
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        verbose=verbose,
    )
    
    start_time = time.time()
    kmeans.train(data)
    end_time = time.time()
    labels = kmeans.predict(data)
    
    elapsed_time = end_time - start_time
    print(f"[PyTorch KMeans] Done in {elapsed_time:.4f} seconds")
    return kmeans.centroids, labels, elapsed_time

def run_fast_pytorch_kmeans(data, k, max_iters=20, seed=42, verbose=False):
    """Run Fast PyTorch KMeans implementation."""
    print(f"\n=== Fast PyTorch KMeans on {data.shape[0]} samples, {k} clusters ===")       
    
    n_samples, n_features = data.shape
    
    # Create Fast PyTorch KMeans object
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, mode='euclidean', verbose=1 if verbose else 0, max_iter=max_iters)
    labels = kmeans.fit_predict(data)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"[Fast PyTorch KMeans] Done in {elapsed_time:.4f} seconds")
    return kmeans.centroids, labels, elapsed_time

def run_faiss_kmeans(data, k, max_iters=20, seed=42, max_points_per_centroid=1_000_000_000, verbose=False, device='cpu'):
    """Run Faiss KMeans implementation."""
    print(f"\n=== Faiss KMeans on {data.shape[0]} samples, {k} clusters ===")       
    
    n_samples, n_features = data.shape
    
    # Create Faiss KMeans object
    print(device)
    kmeans = faiss.Kmeans(
        d=n_features,
        k=k,
        niter=max_iters,
        seed=seed,
        nredo=1,
        gpu=torch.cuda.is_available() and device != 'cpu',
        max_points_per_centroid=max_points_per_centroid,
        verbose=verbose,
    )
    
    start_time = time.time()
    kmeans.train(data)
    end_time = time.time()
    _, labels = kmeans.index.search(data, 1)
    labels = labels.reshape(-1)
    
    elapsed_time = end_time - start_time
    print(f"[Faiss KMeans] Done in {elapsed_time:.4f} seconds")
    return kmeans.centroids, labels, elapsed_time
    
def run_sklearn_kmeans(data, k, max_iters=20, seed=42, verbose=False):
    """Run scikit-learn KMeans implementation."""
    print(f"\n=== scikit-learn KMeans on {data.shape[0]} samples, {k} clusters ===")
    
    # Create scikit-learn KMeans object
    kmeans = SklearnKMeans(
        n_clusters=k,
        max_iter=max_iters,
        random_state=seed,
        init='random',
        n_init=1,
        tol=0,
        verbose=1 if verbose else 0,
    )

    kmeans._tol = -math.inf
    
    start_time = time.time()
    kmeans.fit(data)
    end_time = time.time()
    labels = kmeans.predict(data)
    
    elapsed_time = end_time - start_time
    print(f"[scikit-learn KMeans] Done in {elapsed_time:.4f} seconds")
    return kmeans.cluster_centers_, labels, elapsed_time

def evaluate_clustering(true_labels, predicted_labels, method_name):
    """Evaluate clustering results."""
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f"[{method_name}] Evaluation Metrics:")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    return nmi

def plot_results(benchmarks, results, export_plots=True, device="cpu", random_clusters=False):
    """Plot benchmark results."""
    if not export_plots:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("benchmark_plots", exist_ok=True)
    
    # Set up the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Prepare data for plotting
    datasets = [f"{n_samples/1000:.0f}k-{n_clusters}" for n_samples, n_clusters in benchmarks]
    
    # Format device for title
    device_str = f"({device})" if device != "mps" else "(mps (if supported))"
    
    # Format cluster type for title and filename
    cluster_type = "random" if random_clusters else "structured"
    
    # Plot execution times
    plt.figure(figsize=(14, 8))
    for method in results:
        if 'times' in results[method] and len(results[method]['times']) > 0:
            plt.plot(datasets, results[method]['times'], marker='o', linewidth=2, label=method)
    
    plt.title(f'KMeans Execution Time Comparison {device_str} - {cluster_type} clusters', fontsize=16)
    plt.xlabel('Dataset (samples-clusters)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"benchmark_plots/execution_times_{device}_{cluster_type}.png", dpi=300)
    plt.close()
    
    # Plot NMI scores
    plt.figure(figsize=(14, 8))
    for method in results:
        if 'nmi' in results[method] and len(results[method]['nmi']) > 0:
            plt.plot(datasets, results[method]['nmi'], marker='o', linewidth=2, label=method)
    
    plt.title(f'KMeans Normalized Mutual Information Comparison {device_str} - {cluster_type} clusters', fontsize=16)
    plt.xlabel('Dataset (samples-clusters)', fontsize=14)
    plt.ylabel('NMI Score', fontsize=14)
    plt.ylim(0.5, 1.1)  # Set y-axis limits from 0 to 1.2
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"benchmark_plots/nmi_scores_{device}_{cluster_type}.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to benchmark_plots/ directory (device: {device}, cluster type: {cluster_type})")

def main(
    max_points_per_centroid: int = 1_000_000_000,
    verbose: bool = False,
    do_pytorch_fast_kmeans: bool = False,
    do_sklearn: bool = True,
    do_big_sklearn: bool = False,
    do_faiss: bool = True,
    do_fastkmeans: bool = True,
    device: str = "cpu",
    export_plots: bool = True,
    max_iters: int = 20,
    seed: int = 42,
    n_features: int = 128,
    random_clusters: bool = False
):
    """
    Run KMeans benchmarks with various implementations.
    
    Args:
        max_points_per_centroid: Maximum points per centroid for subsampling
        verbose: Enable verbose output
        do_pytorch_fast_kmeans: Run fast-pytorch-kmeans implementation
        do_sklearn: Run scikit-learn implementation
        do_big_sklearn: Run sklearn even for the big datasets.
        do_faiss: Run Faiss implementation
        do_fastkmeans: Run our FastKMeans implementation
        device: Device to use ('cpu', 'cuda', etc.)
        export_plots: Export plots of benchmark results
        max_iters: Maximum number of iterations
        seed: Random seed
        n_features: Number of features in synthetic data
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Define benchmark configurations
    benchmarks = [
        (10000, 64),      # 10k data points in 64 clusters
        (100000, 256),    # 100k data points into 256 clusters
        (500000, 16),     # 500k into 16 clusters
        (500000, 512),    # 500k into 512 clusters
        (1_000_000, 16),  # 1 million into 16 clusters
        (1_000_000, 1024),  # 1 million into 1024 clusters
        (5_000_000, 2048), # 5 million into 2048 clusters
        (5_000_000, 8192) # 5 million into 8192 clusters
    ]
    
    # Store results for plotting
    results = {
        'Faiss': {'times': [], 'nmi': []},
        'FastKMeans': {'times': [], 'nmi': []},
        'Fast PyTorch KMeans': {'times': [], 'nmi': []},
        'scikit-learn': {'times': [], 'nmi': []}
    }
    
    for n_samples, n_clusters in benchmarks:
        print(f"\n{'='*50}")
        print(f"BENCHMARK: {n_samples} samples, {n_clusters} clusters")
        print(f"{'='*50}")
        
        X, y = generate_synthetic_data(n_samples, n_clusters, n_features, seed, random_clusters)
        
        if do_faiss:
            _, labels_faiss, time_faiss = run_faiss_kmeans(
                X, n_clusters, max_iters, seed, 
                max_points_per_centroid=max_points_per_centroid, 
                verbose=verbose,
                device=device
            )
            nmi = evaluate_clustering(y, labels_faiss, "Faiss KMeans")
            results['Faiss']['times'].append(time_faiss)
            results['Faiss']['nmi'].append(nmi)
        
        if do_fastkmeans:
            _, labels_torch, time_torch = run_fastkmeans(
                X, n_clusters, max_iters, seed, 
                max_points_per_centroid=max_points_per_centroid, 
                verbose=verbose,
                device=device
            )
            nmi = evaluate_clustering(y, labels_torch, "PyTorch KMeans")
            results['FastKMeans']['times'].append(time_torch)
            results['FastKMeans']['nmi'].append(nmi)
        else: del results['FastKMeans']

        if do_pytorch_fast_kmeans and fast_pytorch_kmeans_available and torch.cuda.is_available():
            _, labels_fast_pytorch_kmeans, time_fast_pytorch = run_fast_pytorch_kmeans(
                X, n_clusters, max_iters, seed, verbose=verbose
            )
            nmi = evaluate_clustering(y, labels_fast_pytorch_kmeans, "Fast PyTorch KMeans")
            results['Fast PyTorch KMeans']['times'].append(time_fast_pytorch)
            results['Fast PyTorch KMeans']['nmi'].append(nmi)

        max_sklearn_samples = 500_000 if not do_big_sklearn else 100_000_000
        
        if do_sklearn and n_samples <= max_sklearn_samples:  # Skip for the larger runs because it is _exceedingly_ slow
            _, labels_sklearn, time_sklearn = run_sklearn_kmeans(
                X, n_clusters, max_iters, seed, verbose=verbose
            )
            nmi = evaluate_clustering(y, labels_sklearn, "scikit-learn KMeans")
            results['scikit-learn']['times'].append(time_sklearn)
            results['scikit-learn']['nmi'].append(nmi)
    
    # Plot results
    plot_results(benchmarks, results, export_plots, device, random_clusters)
    
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    typer.run(main)
