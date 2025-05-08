import os, torch, torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from fastkmeans import FastKMeans


def _make_loader(ds, rank, world_size, dl_kwargs):
    if isinstance(ds, IterableDataset):
        return DataLoader(ds, **dl_kwargs)
    else:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        return DataLoader(ds, sampler=sampler, **dl_kwargs)


def _worker_fit(rank, world_size, build_dataset, build_kmeans, dataloader_kwargs, queue):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    ds = build_dataset(rank=rank, world_size=world_size)
    dl = _make_loader(ds, rank, world_size, dataloader_kwargs)

    km = build_kmeans()
    km.fit(dl, device=device)
    if rank == 0:
        queue.put(km.centroids)

    torch.distributed.destroy_process_group()


def _worker_predict(rank, world_size, build_dataset, build_kmeans, dataloader_kwargs, centroids, queue):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    ds = build_dataset(rank=rank, world_size=world_size)
    dl = _make_loader(ds, rank, world_size, dataloader_kwargs)

    km = build_kmeans()
    km.centroids = centroids
    labels, dists, maps = km.predict(dl, device=device)
    if rank == 0:
        queue.put((labels, dists, maps))

    torch.distributed.destroy_process_group()


def distributed_fit(build_dataset,
                    build_kmeans,
                    num_gpus=None,
                    dataloader_kwargs=None):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices found"

    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    queue = mp.Queue()

    mp.spawn(_worker_fit,
             args=(num_gpus, build_dataset, build_kmeans, dataloader_kwargs, queue),
             nprocs=num_gpus,
             join=True)

    return queue.get()


def distributed_predict(build_dataset,
                        build_kmeans,
                        centroids,
                        num_gpus=None,
                        dataloader_kwargs=None):
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    queue = mp.Queue()

    mp.spawn(_worker_predict,
             args=(num_gpus, build_dataset, build_kmeans,
                   dataloader_kwargs, centroids, queue),
             nprocs=num_gpus,
             join=True)

    return queue.get()
