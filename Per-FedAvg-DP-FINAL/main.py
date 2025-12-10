import sys

sys.path.append("data")

import torch
import random
import os
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from rich.progress import track
from utils import get_args, fix_random_seed
from model import get_model
from perfedavg import PerFedAvgClient
from data.utils import get_client_id_indices
import numpy as np
import json

results_file = open("results.txt", "w")

if __name__ == "__main__":
    args = get_args()
    fix_random_seed(args.seed)
    if os.path.isdir("./log") == False:
        os.mkdir("./log")
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    global_model = get_model(args.dataset, device)
    logger = Console(record=True)
    logger.log(f"Arguments:", dict(args._get_kwargs()))
    
    
    clients_4_training, clients_4_eval, client_num_in_total = get_client_id_indices(
        args.dataset
    )

    # build sampling probabilities based on shard sizes 

    # load train stats
    stats_path = os.path.join("data", args.dataset, "all_stats.json")
    with open(stats_path, "r") as f:
        all_stats = json.load(f)["train"]   # only train clients matter here

    # number of training clients
    n = len(clients_4_training)

    # clients sampled per round
    k = args.client_num_per_round      

    # Get shard sizes aligned with clients_4_training
    # shard_sizes[i] corresponds to client_id = clients_4_training[i]
    shard_sizes = np.array(
        [all_stats[f"client {cid}"]["x"] for cid in clients_4_training],
        dtype=float,
    )

    # larger shard with higher probability
    weights = shard_sizes.copy()

    # raw probabilities from weights
    prob_raw = weights / weights.sum()

    # apply lower bound where minimum probability = k / n
    p_min = k / n
    prob_floored = np.maximum(prob_raw, p_min)

    # renormalize so probabilities sums up to 1
    probabilities = prob_floored / prob_floored.sum()


    # init clients
    clients = [
        PerFedAvgClient(
            client_id=client_id,
            alpha=args.alpha,
            beta=args.beta,
            global_model=global_model,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=args.batch_size,
            dataset=args.dataset,
            local_epochs=args.local_epochs,
            valset_ratio=args.valset_ratio,
            logger=logger,
            gpu=args.gpu,
        )
        for client_id in range(client_num_in_total)
    ]
    # training
    logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
    for _ in track(
        range(args.global_epochs), "Training...", console=logger, disable=args.log
    ):
        # where sampling or selection of clients occurs
        # OLD WAY: 
        # selected_clients = random.sample(clients_4_training, args.client_num_per_round)

        # weighted client sampling based on shard size and privacy
        selected_clients = np.random.choice(
        clients_4_training,
        size=args.client_num_per_round,
        replace= True,
        p=probabilities,   # the distribution from above
        )

        model_params_cache = []
        
        # client's local training
        for client_id in selected_clients:
            # train function of perFedAvg called
            serialized_model_params = clients[client_id].train(
                global_model=global_model,
                hessian_free=args.hf,
                eval_while_training=args.eval_while_training,
            )
            # sending local update to the server w_i k+1 T
            model_params_cache.append(serialized_model_params)

        # aggregate and average model parameters
        aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
        SerializationTool.deserialize_model(global_model, aggregated_model_params)
        logger.log("=" * 60)
        
    # evals
    pers_epochs = args.local_epochs if args.pers_epochs == -1 else args.pers_epochs
    logger.log("=" * 20, "EVALUATION", "=" * 20, style="bold blue")
    loss_before = []
    loss_after = []
    acc_before = []
    acc_after = []
    for client_id in track(
        clients_4_eval, "Evaluating...", console=logger, disable=args.log
    ):
        stats = clients[client_id].pers_N_eval(
            global_model=global_model, pers_epochs=pers_epochs,
        )
        loss_before.append(stats["loss_before"])
        loss_after.append(stats["loss_after"])
        acc_before.append(stats["acc_before"])
        acc_after.append(stats["acc_after"])

    # print results

    logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    results_file.write("=" * 20 + " RESULTS " + "=" * 20 + "\n")

    loss_before_val = (sum(loss_before) / len(loss_before))
    acc_before_val = (sum(acc_before) * 100.0 / len(acc_before))
    loss_after_val = (sum(loss_after) / len(loss_after))
    acc_after_val = (sum(acc_after) * 100.0 / len(acc_after))

    logger.log(f"loss_before_pers: {loss_before_val:.4f}")
    logger.log(f"acc_before_pers: {acc_before_val:.2f}%")
    logger.log(f"loss_after_pers: {loss_after_val:.4f}")
    logger.log(f"acc_after_pers: {acc_after_val:.2f}%")

    results_file.write(f"loss_before_pers: {loss_before_val:.4f}\n")
    results_file.write(f"acc_before_pers: {acc_before_val:.2f}%\n")
    results_file.write(f"loss_after_pers: {loss_after_val:.4f}\n")
    results_file.write(f"acc_after_pers: {acc_after_val:.2f}%\n")

    results_file.close()
    
    # SAVE ALL LOGS TO TEXT FILE
    with open("results.txt", "w") as f:
        f.write(logger.export_text())

    if args.log:
        algo = "HF" if args.hf else "FO"
        logger.save_html(
            f"./log/{args.dataset}_{args.client_num_per_round}_{args.global_epochs}_{pers_epochs}_{algo}.html"
        )

