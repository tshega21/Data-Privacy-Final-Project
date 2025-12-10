import sys

sys.path.append("../")
import os
import shutil
import pickle
import numpy as np
import random
import torch
import json
from path import Path
from argparse import ArgumentParser, Namespace
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from dataset import MNISTDataset, CIFARDataset
from collections import Counter
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
#from fedlab.utils.dataset.slicing import noniid_slicing



def noniid_slicing(dataset, num_clients, num_shards):
    """
    Slice dataset into non-IID partitions using variable-size shards.
    """

    np.random.seed(21)

    total_samples = len(dataset)

    raw = np.random.rand(num_shards)
    shard_sizes = (raw / raw.sum() * total_samples).astype(int)

    # Fix rounding mismatch
    diff = total_samples - shard_sizes.sum()
    shard_sizes[0] += diff  # adjust the first shard so total matches exactly


    # Sort data by label 
    labels = np.array(dataset.targets)
    idxs = np.arange(total_samples)
    sorted_idxs = idxs[np.argsort(labels)]

    # Slice sorted indices into variable-sized shards 
    shards = []
    ptr = 0
    for size in shard_sizes:
        shards.append(sorted_idxs[ptr:ptr+size])
        ptr += size

    # Assign shards to clients
    shards_per_client = num_shards // num_clients
   

    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}
    shard_ids = np.arange(num_shards)

    for client in range(num_clients):
        chosen = np.random.choice(shard_ids, shards_per_client, replace=False)
        shard_ids = np.setdiff1d(shard_ids, chosen)

        for s in chosen:
            dict_users[client] = np.concatenate((dict_users[client], shards[s]))

    return dict_users






CURRENT_DIR = Path(__file__).parent.abspath()

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar": (CIFAR10, CIFARDataset),
}


MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
}

STD = {
    "mnist": (0.3015,),
    "cifar": (0.2023, 0.1994, 0.2010),
}


def preprocess(args: Namespace) -> None:
    dataset_dir = CURRENT_DIR / args.dataset
    pickles_dir = CURRENT_DIR / args.dataset / "pickles"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # splits def 200 clients into 180 training and 20 testing clients
    num_train_clients = int(args.client_num_in_total * args.fraction)
    num_test_clients = args.client_num_in_total - num_train_clients

    transform = transforms.Compose(
        [transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),]
    )
    target_transform = None
    trainset_stats = {}
    testset_stats = {}

    if not os.path.isdir(CURRENT_DIR / args.dataset):
        os.mkdir(CURRENT_DIR / args.dataset)
    if os.path.isdir(pickles_dir):
        shutil.rmtree(pickles_dir)
    os.mkdir(f"{pickles_dir}")

    ori_dataset, target_dataset = DATASET[args.dataset]
    trainset = ori_dataset(
        dataset_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    testset = ori_dataset(dataset_dir, train=False, transform=transforms.ToTensor())

    num_classes = 10 if args.classes <= 0 else args.classes
    all_trainsets, trainset_stats = randomly_alloc_classes(
        ori_dataset=trainset,
        target_dataset=target_dataset,
        num_clients=num_train_clients,
        num_classes=num_classes,
        transform=transform,
        target_transform=target_transform,
    )
    all_testsets, testset_stats = randomly_alloc_classes(
        ori_dataset=testset,
        target_dataset=target_dataset,
        num_clients=num_test_clients,
        num_classes=num_classes,
        transform=transform,
        target_transform=target_transform,
    )

    all_datasets = all_trainsets + all_testsets

    for client_id, dataset in enumerate(all_datasets):
        with open(pickles_dir / str(client_id) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
            
    with open(pickles_dir / "seperation.pkl", "wb") as f:
        pickle.dump(
            {
                "train": [i for i in range(num_train_clients)],
                "test": [i for i in range(num_train_clients, args.client_num_in_total)],
                "total": args.client_num_in_total,
            },
            f,
        )
    with open(dataset_dir / "all_stats.json", "w") as f:
        json.dump({"train": trainset_stats, "test": testset_stats}, f)


def randomly_alloc_classes(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    dict_users = noniid_slicing(ori_dataset, num_clients, num_clients * num_classes)
    stats = {}
    for i, indices in dict_users.items():
        targets_numpy = np.array(ori_dataset.targets)
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    
        
    for indices in dict_users.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )

    return datasets, stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar"], default="mnist",
    )
    parser.add_argument("--client_num_in_total", type=int, default=200)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=2,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)
