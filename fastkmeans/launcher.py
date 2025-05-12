import os
import socket

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from fastkmeans import FastKMeans


mp.set_start_method("spawn", force=True)


def _setup_master_env():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()


def _make_loader(ds, rank, world_size, dl_kwargs):
    if isinstance(ds, IterableDataset):
        return DataLoader(ds, **dl_kwargs)
    else:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        return DataLoader(ds, sampler=sampler, **dl_kwargs)


def _worker(
    rank: int,
    world_size: int,
    build_dataset,
    build_kmeans,
    dataloader_kwargs: dict,
    queue: mp.Queue,
    mode: str,
    centroids=None,
):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    ds = build_dataset()
    dl = _make_loader(ds, rank, world_size, dataloader_kwargs)

    km = build_kmeans()

    if mode == "fit":
        centroids = km.fit(dataloader=dl, device=device)
        if rank == 0:
            queue.put(centroids)
    elif mode == "predict":
        if centroids is None:
            raise ValueError("Centroids must be provided for prediction.")
        labels, distances, mappings = km.predict(centroids=centroids, dataloader=dl, device=device)
        if rank == 0:
            queue.put((labels, distances, mappings))

    torch.distributed.destroy_process_group()


def distributed_fit(build_datasets,
                    build_kmeans,
                    num_gpus,
                    dataloader_kwargs):
    manager = mp.Manager()
    queue = manager.Queue()

    _setup_master_env()

    procs = []
    for rank, build_dataset in enumerate(build_datasets):
        p = mp.Process(
            target=_worker,
            args=(rank, num_gpus, build_dataset, build_kmeans, dataloader_kwargs, queue, "fit"),
        )
        p.start()
        procs.append(p)

    centroids = queue.get()

    for p in procs:
        p.join()

    manager.shutdown()

    return centroids


def distributed_predict(build_datasets,
                        build_kmeans,
                        centroids,
                        num_gpus,
                        dataloader_kwargs):
    manager = mp.Manager()
    queue = manager.Queue()

    _setup_master_env()
    
    procs = []
    for rank, build_dataset in enumerate(build_datasets):
        p = mp.Process(
            target=_worker,
            args=(rank, num_gpus, build_dataset, build_kmeans, dataloader_kwargs, queue, "predict", centroids),
        )
        p.start()
        procs.append(p)

    labels, distances, mappings = queue.get()

    for p in procs:
        p.join()

    manager.shutdown()

    return labels, distances, mappings
