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


def _get_random_data(dataloader: DataLoader, n_samples: int, rng: torch.Generator) -> torch.Tensor:
    """
    Returns a random sample of data from the dataloader on the current rank.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to sample from (expected to be for the current rank).
    n_samples : int
        Desired number of samples.
    rng : torch.Generator
        PyTorch random number generator.

    Returns
    -------
    torch.Tensor
        Randomly sampled data from this rank.
    """
    chunks, total = [], 0
    for batch in dataloader:
        data = batch[-1] if isinstance(batch, (list, tuple)) else batch
        chunks.append(data)
        total += data.shape[0]
        if total >= n_samples:
            break
    chunks = torch.cat(chunks, dim=0)
    random_indices = torch.randperm(chunks.shape[0], generator=rng)[:n_samples]
    chunks = chunks[random_indices]
    return chunks


def _assign_lloyd(
    data_chunk: torch.Tensor,
    data_chunk_norms: torch.Tensor,
    centroids: torch.Tensor,
    centroid_norms: torch.Tensor,
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
        best_dist[improved_mask] = local_min_vals[improved_mask].float()
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

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        if use_triton and not HAS_TRITON:
            raise ValueError("Triton is not available. Please install Triton and try again.")
        self.use_triton = use_triton

    @torch.no_grad()
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
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        random_data = _get_random_data(dataloader, self.k, self.rng).to(device=device, dtype=self.dtype)

        if dist.is_initialized():
            actual_length = random_data.size(0)
            if actual_length < self.k:
                padding = self.k - actual_length
                random_data = torch.cat([random_data, torch.empty(padding, self.d, device=device, dtype=self.dtype)])

            length = torch.tensor([actual_length], device=device, dtype=torch.int64)

            if rank == 0:
                gathered_data = [torch.empty_like(random_data) for _ in range(world_size)]
                all_lengths = [torch.empty_like(length) for _ in range(world_size)]
                dist.gather(random_data, gathered_data, dst=0)
                dist.gather(length, all_lengths, dst=0)
            else:
                dist.gather(random_data, dst=0)
                dist.gather(length, dst=0)

            dist.barrier()

            if rank == 0:
                candidates = torch.cat([gathered_data[i][:all_lengths[i].item()] for i in range(world_size)], dim=0)
                random_indices = torch.randperm(candidates.size(0), generator=self.rng)[:self.k].to(device)
                centroids = candidates[random_indices].contiguous()
            else:
                centroids = torch.empty(self.k, self.d, device=device, dtype=self.dtype)

            dist.broadcast(centroids, src=0)
        else:
            centroids = random_data[:self.k].to(device)

        variances = torch.var(centroids, dim=0)
        _tol = torch.mean(variances) * self.tol

        centroid_norms = torch.empty((self.k,), device=device, dtype=self.dtype)
        best_dist, best_ids = None, None

        for iteration in range(self.niter):
            iteration_start_time = time.time()

            centroid_norms.copy_((centroids ** 2).sum(dim=1))

            cluster_fused_data = torch.zeros((self.k, self.d + 1), device=device, dtype=torch.float32)

            with tqdm(desc=f"Fitting ({iteration + 1}th iteration)", disable=rank != 0, total=len(dataloader)) as pbar:
                for batch in dataloader:
                    data_chunk = batch[-1] if isinstance(batch, (list, tuple)) else batch
                    data_chunk = data_chunk.to(device, dtype=self.dtype, non_blocking=True)
                    data_chunk_norms = (data_chunk ** 2).sum(dim=1)

                    batch_size = data_chunk.size(0)
                    if best_dist is None or len(best_dist) != batch_size:
                        best_dist = torch.full((batch_size,), float("inf"), device=device, dtype=torch.float32)
                    else:
                        best_dist.fill_(float("inf"))
                    if best_ids is None or len(best_ids) != batch_size:
                        best_ids = torch.zeros((batch_size,), device=device, dtype=torch.long)
                    else:
                        best_ids.fill_(0)

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
                            centroid_norms=centroid_norms,
                            best_dist=best_dist,
                            best_ids=best_ids,
                            chunk_size_centroids=self.chunk_size_centroids,
                        )

                    cluster_fused_data[:, :-1].index_add_(0, best_ids, data_chunk.float())
                    cluster_fused_data[:, -1].index_add_(0, best_ids, torch.ones_like(best_ids, dtype=torch.float32))

                    pbar.update(1)

            if dist.is_initialized():
                dist.reduce(cluster_fused_data, dst=0, op=dist.ReduceOp.SUM)

            if not dist.is_initialized() or rank == 0:
                sums = cluster_fused_data[:, :-1]
                counts = cluster_fused_data[:, -1]
                new_centroids = torch.zeros_like(sums, dtype=self.dtype)
                mask = counts > 0
                new_centroids[mask] = (sums[mask] / counts[mask].unsqueeze(1)).to(self.dtype)

                empty_ids = (~mask).nonzero(as_tuple=True)[0]
                if len(empty_ids) > 0:
                    random_data = _get_random_data(dataloader, len(empty_ids), self.rng).to(device=device, dtype=self.dtype)
                    new_centroids[empty_ids] = random_data

                shift = torch.norm(new_centroids - centroids, dim=1, dtype=torch.float).sum()
                centroids.copy_(new_centroids)
            else:
                shift = torch.tensor(0.0, device=device, dtype=torch.float32)

            if dist.is_initialized():
                dist.broadcast(centroids, src=0)
                dist.broadcast(shift, src=0)

            iteration_time = time.time() - iteration_start_time
            if self.verbose and rank == 0:
                print(
                    f"Iteration {iteration + 1}/{self.niter} took {iteration_time:.4f}s, shift: {shift.item():.6f}"
                )

            if shift.item() < _tol:
                if self.verbose and rank == 0:
                    print(f"Converged after {iteration + 1} iterations (shift: {shift.item():.6f} < tol: {_tol})")
                break

        if rank == 0:
            return centroids.cpu().contiguous()

    @torch.no_grad()
    def predict(self, centroids: np.ndarray, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, list]:
        """
        Predicts the cluster labels for the data in the dataloader.

        Parameters
        ----------
        centroids : np.ndarray
            Centroids to use for prediction.
        dataloader : DataLoader
            Dataloader to predict the labels for.
        device : torch.device
            Device to use for computation.

        Returns
        -------
        tuple[np.ndarray, list]
            Tuple containing the predicted labels and a list of mappings.
        """
        centroids_torch = torch.as_tensor(centroids, device=device, dtype=self.dtype)
        centroid_norms = (centroids_torch**2).sum(dim=1)

        mappings = []
        labels = []
        distances = []

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        best_dist, best_ids = None, None

        with tqdm(desc="Prediction", disable=rank != 0, total=len(dataloader)) as pbar:
            for batch in dataloader:
                data_chunk = batch[-1] if isinstance(batch, (list, tuple)) else batch
                data_chunk = data_chunk.to(device=device, dtype=self.dtype, non_blocking=True)
                data_chunk_norms = (data_chunk**2).sum(dim=1)

                batch_size = data_chunk.size(0)
                if best_dist is None or len(best_dist) != batch_size:
                    best_dist = torch.full((batch_size,), float("inf"), device=device, dtype=torch.float32)
                else:
                    best_dist.fill_(float("inf"))
                if best_ids is None or len(best_ids) != batch_size:
                    best_ids = torch.zeros((batch_size,), device=device, dtype=torch.long)

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

                if isinstance(batch, (list, tuple)) and len(batch) > 1:
                    for mapping in zip(*batch[:-1]):
                        mappings.append(mapping)

                pbar.update(1)

        labels = torch.cat(labels, dim=0)
        distances = torch.cat(distances, dim=0)

        if dist.is_initialized():
            gloo_group = dist.new_group(backend="gloo", ranks=list(range(world_size)))

            length = torch.tensor([labels.numel()], dtype=torch.int64)
            all_lengths = [torch.empty_like(length) for _ in range(world_size)]
            dist.all_gather(all_lengths, length, group=gloo_group)

            max_length = max([l.item() for l in all_lengths])

            if max_length > length.item():
                pad = max_length - labels.size(0)
                labels = torch.cat([labels, torch.empty(pad, dtype=torch.long)])
                distances = torch.cat([distances, torch.empty(pad, dtype=torch.float32)])

            if rank == 0:
                gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
                gathered_distances = [torch.empty_like(distances) for _ in range(world_size)]
                gathered_mappings = [None] * world_size
                dist.gather(labels, gathered_labels, dst=0, group=gloo_group)
                dist.gather(distances, gathered_distances, dst=0, group=gloo_group)
                dist.gather_object(mappings, gathered_mappings, dst=0, group=gloo_group)
            else:
                dist.gather(labels, dst=0, group=gloo_group)
                dist.gather(distances, dst=0, group=gloo_group)
                dist.gather_object(mappings, dst=0, group=gloo_group)

            dist.barrier(group=gloo_group)

            if rank == 0:
                labels = torch.cat([gathered_labels[i][:all_lengths[i].item()] for i in range(world_size)], dim=0)
                distances = torch.cat([gathered_distances[i][:all_lengths[i].item()] for i in range(world_size)], dim=0)
                mappings = sum(gathered_mappings, [])

            dist.destroy_process_group(gloo_group)

        if rank == 0:
            return labels, distances, mappings
        else:
            return None, None, None
