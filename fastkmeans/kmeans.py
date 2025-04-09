import time

import torch
import numpy as np

def _get_device(preset: str = None):
    if preset: return preset
    if torch.cuda.is_available(): return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

@torch.inference_mode()
def _kmeans_torch_double_chunked(
    data: torch.Tensor,
    data_norms: torch.Tensor,
    k: int,
    max_iters: int = 25,
    tol: float = 1e-8,
    device: str = None,
    dtype: torch.dtype = None,
    chunk_size_data: int = 50_000,
    chunk_size_centroids: int = 10_000,
    max_points_per_centroid: int = 256,
    verbose: bool = False,
):
    """
    An efficient kmeans implementation that minimises OOM risks on modern hardware by using conversative double chunking.

    Returns
    -------
    centroids_cpu : torch.Tensor, shape (k, n_features), float32
    labels_cpu    : torch.Tensor, shape (n_samples_used,), long
        Where n_samples_used can be smaller than the original if subsampling occurred.
    """
    n_samples_original, n_features = data.shape
    n_samples = n_samples_original

    if max_points_per_centroid is not None and n_samples > k * max_points_per_centroid:
        target_n_samples = k * max_points_per_centroid
        perm = torch.randperm(n_samples, device=data.device)
        indices = perm[:target_n_samples]
        data = data[indices]
        data_norms = data_norms[indices]
        n_samples = target_n_samples
        del perm, indices

    if n_samples < k:
        raise ValueError(f"Number of training points ({n_samples}) is less than k ({k}).")

    if dtype is None: dtype = torch.float16 if device == 'cuda' else torch.float32

    # centroid init -- random is the only supported init
    rand_indices = torch.randperm(n_samples)[:k]
    centroids = data[rand_indices].clone().to(device=device, dtype=dtype)
    prev_centroids = centroids.clone()

    labels = torch.empty(n_samples, dtype=torch.long, device='cpu')  # Keep labels on CPU


    for iteration in range(max_iters):
        iteration_start_time = time.time()
        
        centroid_norms = (centroids ** 2).sum(dim=1)
        cluster_sums = torch.zeros((k, n_features), device=device, dtype=torch.float32)
        cluster_counts = torch.zeros((k,), device=device, dtype=torch.float32)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + chunk_size_data, n_samples)

            data_chunk = data[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            data_chunk_norms = data_norms[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            batch_size = data_chunk.size(0)

            best_dist = torch.full((batch_size,), float('inf'), device=device, dtype=dtype)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.long)

            c_start = 0
            while c_start < k:
                c_end = min(c_start + chunk_size_centroids, k)
                centroid_chunk = centroids[c_start:c_end]
                centroid_chunk_norms = centroid_norms[c_start:c_end]

                dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                dist_chunk = dist_chunk.addmm_(
                    data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0
                )

                local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                improved_mask = local_min_vals < best_dist
                best_dist[improved_mask] = local_min_vals[improved_mask]
                best_ids[improved_mask] = (c_start + local_min_ids[improved_mask])

                c_start = c_end

            cluster_sums.index_add_(0, best_ids, data_chunk.float())
            cluster_counts.index_add_(0, best_ids, torch.ones_like(best_ids, dtype=torch.float32))

            labels[start_idx:end_idx] = best_ids.to('cpu', non_blocking=True)
            start_idx = end_idx

        new_centroids = torch.zeros_like(centroids, device=device, dtype=torch.float32)
        non_empty = (cluster_counts > 0)
        new_centroids[non_empty] = (
            cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)
        )

        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty_ids) > 0:
            reinit_indices = torch.randint(0, n_samples, (len(empty_ids),), device='cpu')
            random_data = data[reinit_indices].to(device=device, dtype=torch.float32)
            new_centroids[empty_ids] = random_data

        new_centroids = new_centroids.to(dtype=dtype)

        shift = torch.norm(new_centroids - prev_centroids.to(new_centroids.device), dim=1).sum().item()
        centroids = new_centroids

        prev_centroids = centroids.clone()
        
        iteration_time = time.time() - iteration_start_time
        if verbose: print(f"Iteration {iteration+1}/{max_iters} took {iteration_time:.4f}s, total time: {time.time() - iteration_start_time + iteration_time:.4f}s, shift: {shift:.6f}")
        
        if shift < tol:
            if verbose: print(f"Converged after {iteration+1} iterations (shift: {shift:.6f} < tol: {tol})")
            break

    centroids_cpu = centroids.to('cpu', dtype=torch.float32)
    return centroids_cpu, labels


class FastKMeans:
    """
    A drop-in replacement for Faiss's Kmeans API, implemented with PyTorch
    double-chunked KMeans under the hood.

    Parameters
    ----------
    d  : int
        Dimensionality of the input features (n_features).
    k  : int
        Number of clusters.
    niter : int, default=20
        Maximum number of iterations.
    tol : float, default=1e-4
        Stopping threshold for centroid movement.
    gpu : bool, default=True
        Whether to force GPU usage if available. If False, CPU is used.
    seed : int, default=0
        Random seed for centroid initialization and (if needed) subsampling.
    max_points_per_centroid : int, optional, default=1_000_000_000
        If n_samples > k * max_points_per_centroid, the data will be subsampled to exactly
        k * max_points_per_centroid points before clustering.
    chunk_size_data : int, default=50_000
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=10_000
        Chunk size along the centroid dimension for assignment/update steps.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        gpu: bool = True,
        seed: int = 0,
        max_points_per_centroid: int = 256,
        chunk_size_data: int = 50_000,
        chunk_size_centroids: int = 10_000,
        device: str = None,
        dtype: torch.dtype = None,
        pin_gpu_memory: bool = True,
        verbose: bool = False,
        nredo: int = 1, # for compatibility only
    ):
        self.d = d
        self.k = k
        self.niter = niter
        self.tol = tol
        self.gpu = gpu
        self.seed = seed
        self.max_points_per_centroid = max_points_per_centroid
        self.chunk_size_data = chunk_size_data
        self.chunk_size_centroids = chunk_size_centroids
        self.centroids = None
        if device not in [None, 'cuda'] and self.gpu: print("Warning: device is set to 'cuda' but gpu is True, ignoring 'device' argument and setting it to 'cuda'!")
        self.device = 'cuda' if self.gpu else device
        self.dtype = dtype
        self.pin_gpu_memory = pin_gpu_memory
        if nredo != 1: raise ValueError("nredo must be 1, redos not currently supported")
        self.verbose = verbose

    def train(self, data: np.ndarray):
        """
        Trains (fits) the KMeans model on the given data and sets `self.centroids`. Designed to mimic faiss's `train()` method.
        
        Parameters
        ----------
        data : np.ndarray of shape (n_samples, d), float32
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        # Move data to PyTorch CPU Tensor
        data_torch = torch.from_numpy(data)
        data_norms_torch = (data_torch ** 2).sum(dim=1)

        device = _get_device(self.device)
        if device == 'cuda' and self.pin_gpu_memory:
            data_torch = data_torch.pin_memory()
            data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch,
            k=self.k,
            max_iters=self.niter,
            tol=self.tol,
            device=device,
            dtype=self.dtype,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            max_points_per_centroid=self.max_points_per_centroid,
            verbose=self.verbose,
        )
        self.centroids = centroids.numpy()

    def fit(self, data: np.ndarray):
        """
        Same as train(), included for interface similarity with scikit-learn's `fit()`.
        """
        self.train(data)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Assigns each data point to the nearest centroid for even more compatibility with scikit-learn's `predict()`, which is what cool libraries do.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,), int64
        """
        if self.centroids is None:
            raise RuntimeError("Must call train() or fit() before predict().")

        data_torch = torch.from_numpy(data)
        data_norms_torch = (data_torch ** 2).sum(dim=1)

        # We'll do a chunked assignment pass, similar to the main loop, but no centroid updates
        centroids_torch = torch.from_numpy(self.centroids)
        device = centroids_torch.device.type
        if device == 'cpu' and self.gpu and torch.cuda.is_available():
            device = 'cuda'  # If user asked for GPU, put centroids there

        centroids_torch = centroids_torch.to(device=device, dtype=torch.float32)
        centroid_norms = (centroids_torch ** 2).sum(dim=1)

        n_samples = data_torch.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device='cpu')

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + self.chunk_size_data, n_samples)
            data_chunk = data_torch[start_idx:end_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            data_chunk_norms = data_norms_torch[start_idx:end_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            batch_size = data_chunk.size(0)

            best_dist = torch.full((batch_size,), float('inf'), device=device, dtype=torch.float32)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.long)

            c_start = 0
            k = centroids_torch.shape[0]
            while c_start < k:
                c_end = min(c_start + self.chunk_size_centroids, k)
                centroid_chunk = centroids_torch[c_start:c_end]
                centroid_chunk_norms = centroid_norms[c_start:c_end]
                
                dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                dist_chunk = dist_chunk.addmm_(
                    data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0
                )

                local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                improved_mask = local_min_vals < best_dist
                best_dist[improved_mask] = local_min_vals[improved_mask]
                best_ids[improved_mask] = (c_start + local_min_ids[improved_mask])
                c_start = c_end

            labels[start_idx:end_idx] = best_ids.to('cpu')
            start_idx = end_idx

        return labels.numpy()

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Chains fit and predict, once again inspired by the great scikit-learn.
        """
        self.fit(data)
        return self.predict(data)
