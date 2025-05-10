import time

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from fastkmeans.triton_kernels import triton_kmeans

    HAS_TRITON = True
except ImportError:
    triton_kmeans = None
    HAS_TRITON = False


def _get_random_features(dataloader: DataLoader, n_samples: int) -> torch.Tensor:
    """
    Returns a random sample of features from the dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to sample from.
    n_samples : int
        Number of samples to return.

    Returns
    -------
    torch.Tensor
        Randomly sampled features.
    """
    chunks, total = [], 0
    for _, _, features in dataloader:
        chunks.append(features)
        total += features.shape[0]
        if total >= n_samples * 10:
            break
    chunks = torch.cat(chunks, dim=0)
    return chunks[torch.randperm(chunks.shape[0])[:n_samples]]


def _assign_lloyd(
    data_chunk: torch.Tensor,
    data_chunk_norms: torch.Tensor,
    centroids: torch.Tensor,
    centroids_norms: torch.Tensor,
    best_dist: torch.Tensor,
    best_ids: torch.Tensor,
    chunk_size_centroids: int,
):
    c_start = 0
    while c_start < centroids.size(0):
        c_end = min(c_start + chunk_size_centroids, centroids.size(0))
        centroid_chunk = centroids[c_start:c_end]
        centroid_chunk_norms = centroid_norms[c_start:c_end]

        dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
        dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

        local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
        improved_mask = local_min_vals < best_dist
        best_dist[improved_mask] = local_min_vals[improved_mask]
        best_ids[improved_mask] = c_start + local_min_ids[improved_mask]

        c_start = c_end


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
    seed : int, default=0
        Random seed for centroid initialization and (if needed) subsampling.
    chunk_size_centroids : int, default=10,240
        Chunk size along the centroid dimension for assignment/update steps.
    dtype : torch.dtype, default=torch.half
        Data type for the input features and centroids.
    verbose : bool, default=False
        If True, print progress information.
    use_triton : bool, default=True
       Use the fast Triton backend for the assignment/update steps.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        seed: int = 0,
        chunk_size_centroids: int = 10_240,
        dtype: torch.dtype = torch.half,
        verbose: bool = False,
        use_triton: bool = True,
    ):
        self.d = d
        self.k = k
        self.niter = niter
        self.tol = tol
        self.seed = seed
        self.chunk_size_centroids = chunk_size_centroids
        self.dtype = dtype
        self.verbose = verbose

        if seed is not None:
            torch.manual_seed(seed)

        if use_triton and not HAS_TRITON:
            raise ValueError("Triton is not available. Please install Triton and try again.")
        self.use_triton = use_triton

        self.centroids = None

    @torch.inference_mode()
    def fit(self, dataloader: DataLoader, device: torch.device):
        """
        Fits the KMeans model to the data.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader to fit the model to.
        device : torch.device
            Device to use for computation.
        """
        torch.backends.cuda.matmul.allow_tf32 = True

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if rank == 0:
            init = _get_random_features(dataloader, self.k).to(device=device, dtype=self.dtype)
            centroids = init.contiguous()
        else:
            centroids = torch.empty((self.k, self.d), device=device, dtype=self.dtype)
        prev_centroids = centroids.clone()

        if dist.is_initialized():
            dist.broadcast(centroids, src=0)

        copy_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)

        centroid_norms = torch.empty((self.k,), device=device, dtype=self.dtype)
        best_dist, best_ids = None, None

        for iteration in range(self.niter):
            iteration_start_time = time.time()

            centroid_norms.copy_((centroids ** 2).sum(dim=1))

            cluster_sums = torch.zeros((self.k, self.d), device=device, dtype=torch.float32)
            cluster_counts = torch.zeros((self.k,), device=device, dtype=torch.float32)

            with tqdm(desc="Processing batches", disable=rank != 0) as pbar:
                for batch in dataloader:
                    with torch.cuda.stream(copy_stream):
                        if isinstance(batch, tuple):
                            data_chunk = batch[-1]
                        else:
                            data_chunk = batch
                            
                        data_chunk = data.to(device, dtype=self.dtype, non_blocking=True)
                        data_chunk_norms = (data_chunk ** 2).sum(dim=1)
                    copy_stream.synchronize()

                    batch_size = data_chunk.size(0)

                    best_dist = torch.full((batch_size,), float("inf"), device=device, dtype=torch.float32)
                    best_ids = torch.empty((batch_size,), device=device, dtype=torch.long)

                    with torch.cuda.stream(compute_stream):
                        if self.use_triton:
                            triton_kmeans(
                                data_chunk=data_chunk,
                                data_chunk_norms=data_chunk_norms,
                                centroids=centroids,
                                centroids_sqnorm=centroid_norms,
                                best_dist=best_dist,
                                best_ids=best_ids,
                            )
                        else:
                            _assign_lloyd(
                                data_chunk=data_chunk,
                                data_chunk_norms=data_chunk_norms,
                                centroids=centroids,
                                centroids_norms=centroid_norms,
                                best_dist=best_dist,
                                best_ids=best_ids,
                                chunk_size_centroids=self.chunk_size_centroids,
                            )

                        cluster_sums.index_add_(0, best_ids, data_chunk.float())
                        cluster_counts.index_add_(0, best_ids, torch.ones_like(best_ids, dtype=torch.float32))

                    pbar.update(1)

            if dist.is_initialized():
                dist.all_reduce(cluster_sums, op=dist.ReduceOp.SUM, async_op=True)
                dist.all_reduce(cluster_counts, op=dist.ReduceOp.SUM, async_op=True)

            if rank == 0:
                new_centroids = torch.zeros_like(centroids)
                non_empty = cluster_counts > 0
                new_centroids[non_empty] = (cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)).to(dtype=self.dtype)

                empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
                if len(empty_ids) > 0:
                    random_data = _get_random_features(dataloader, len(empty_ids)).to(device=device, dtype=self.dtype)
                    new_centroids[empty_ids] = random_data

                shift = torch.norm(new_centroids - prev_centroids, dim=1).sum()
                centroids.copy_(new_centroids)
                prev_centroids = centroids.clone()
            else:
                shift = torch.zeros((1,), device=device, dtype=self.dtype)

            if dist.is_initialized():
                dist.broadcast(centroids, src=0)
                dist.broadcast(shift, src=0)

            shift = shift.item()

            iteration_time = time.time() - iteration_start_time
            if self.verbose and rank == 0:
                print(
                    f"Iteration {iteration + 1}/{self.niter} took {iteration_time:.4f}s, total time: {time.time() - iteration_start_time + iteration_time:.4f}s, shift: {shift:.6f}"
                )

            if shift < self.tol:
                if self.verbose and rank == 0:
                    print(f"Converged after {iteration + 1} iterations (shift: {shift:.6f} < tol: {self.tol})")
                break

        if rank == 0:
            self.centroids = centroids.cpu().contiguous().numpy()

    @torch.inference_mode()
    def predict(self, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, list]:
        """
        Predicts the cluster labels for the data in the dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader to predict the labels for.
        device : torch.device
            Device to use for computation.

        Returns
        -------
        tuple[np.ndarray, list]
            Tuple containing the predicted labels and a list of mappings.
        """
        if self.centroids is None:
            raise RuntimeError("Must call train() or fit() before predict().")

        centroids_torch = torch.as_tensor(self.centroids, device=device, dtype=self.dtype)
        centroid_norms = (centroids_torch**2).sum(dim=1)

        mappings = []
        labels = []
        distances = []

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        copy_stream = torch.cuda.Stream(device)
        compute_stream = torch.cuda.Stream(device)

        with tqdm(desc="Processing batches", disable=rank != 0) as pbar:
            for batch in dataloader:
                with torch.cuda.stream(copy_stream):
                    if isinstance(batch, tuple):
                        data_chunk = batch[-1]
                    else:
                        data_chunk = batch

                    data_chunk = data_chunk.to(device=device, dtype=self.dtype, non_blocking=True)
                    data_chunk_norms = (data_chunk**2).sum(dim=1)
                copy_stream.synchronize()

                batch_size = data_chunk.size(0)

                best_dist = torch.full((batch_size,), float("inf"), device=device, dtype=torch.float32)
                best_ids = torch.empty((batch_size,), device=device, dtype=torch.long)

                with torch.cuda.stream(compute_stream):
                    if self.use_triton:
                        triton_kmeans(
                            data_chunk,
                            data_chunk_norms,
                            centroids_torch,
                            centroid_norms,
                            best_ids,
                        )
                    else:
                        _assign_lloyd(
                            data_chunk,
                            data_chunk_norms,
                            centroids_torch,
                            centroid_norms,
                            best_dist,
                            best_ids,
                            chunk_size_centroids=self.chunk_size_centroids,
                        )
                    
                distances.append(best_dist.cpu())
                labels.append(best_ids.cpu())

                if isinstance(batch, tuple):
                    for mapping in zip(*batch[:-1]):
                        mappings.append(mapping)

                pbar.update(1)

        labels = torch.cat(labels, dim=0).numpy()
        distances = torch.cat(distances, dim=0).numpy()

        if dist.is_initialized():
            pack = {"labels": labels, "distances": distances, "mappings": mappings}
            gathered = [None] * world_size
            dist.all_gather_object(gathered, pack)

            if rank == 0:
                labels = np.concatenate([g["labels"] for g in gathered], axis=0)
                distances = np.concatenate([g["distances"] for g in gathered], axis=0)
                mappings = sum([g["mappings"] for g in gathered], [])
                return labels, distances, mappings
            else:
                return None, None, None
        else:
            return labels, distances, mappings
