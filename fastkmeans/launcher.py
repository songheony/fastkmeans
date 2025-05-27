import os
import socket

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader


def _setup_master_env():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()

def _worker(
    rank: int,
    world_size: int,
    build_dataset,
    build_kmeans,
    dataloader_kwargs: dict,
    mode: str,
    centroids=None,
):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        ds = build_dataset()
        dl = DataLoader(ds, **dataloader_kwargs)

        km = build_kmeans()

        if mode == "fit":
            centroids = km.fit(dataloader=dl, device=device)
            if rank == 0:
                return centroids
        elif mode == "predict":
            if centroids is None:
                raise ValueError("Centroids must be provided for prediction.")
            labels, distances, mappings = km.predict(centroids=centroids, dataloader=dl, device=device)
            if rank == 0:
                return labels, distances, mappings
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def distributed_fit(ctx,
                    build_datasets,
                    build_kmeans,
                    num_gpus,
                    dataloader_kwargs):
    _setup_master_env()

    procs = []
    try:
        for rank in range(1, len(build_datasets)):
            p = ctx.Process(
                target=_worker,
                args=(rank, num_gpus, build_datasets[rank], build_kmeans, dataloader_kwargs, "fit"),
            )
            p.start()
            procs.append(p)

        centroids = _worker(
            rank=0,
            world_size=num_gpus,
            build_dataset=build_datasets[0],
            build_kmeans=build_kmeans,
            dataloader_kwargs=dataloader_kwargs,
            mode="fit",
        )

        for p in procs:
            p.join()
    except FileNotFoundError:
        pass
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()

    return centroids


def distributed_predict(ctx,
                        build_datasets,
                        build_kmeans,
                        centroids,
                        num_gpus,
                        dataloader_kwargs):
    _setup_master_env()
    
    procs = []
    try:
        for rank in range(1, len(build_datasets)):
            p = ctx.Process(
                target=_worker,
                args=(rank, num_gpus, build_datasets[rank], build_kmeans, dataloader_kwargs, "predict", centroids),
            )
            p.start()
            procs.append(p)

        labels, distances, mappings = _worker(
            rank=0,
            world_size=num_gpus,
            build_dataset=build_datasets[0],
            build_kmeans=build_kmeans,
            dataloader_kwargs=dataloader_kwargs,
            mode="predict",
            centroids=centroids,
        )

        for p in procs:
            p.join()
    except FileNotFoundError:
        pass
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()

    return labels, distances, mappings
