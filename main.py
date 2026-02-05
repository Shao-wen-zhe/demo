from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch_geometric.graphgym.loader import create_loader
from dataset_loader import DataLoader
from model import Model, APPNPModule
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                              set_cfg, load_cfg,
                                             makedirs_rm_exist)
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup, set_seed, random_planetoid_splits
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import optuna

def get_args():
    # Training settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='amazon-ratings',
                            choices=['cora', 'citeseer', 'pubmed','roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                    'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                    'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                    'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin','computers','photo'])
    # model
    parser.add_argument('--model', type=str, default='GateMambaGCN',
                        choices=['GateMambaGCN', 'GatedGCNLayer','GENConv','ResNet', 'APPNP',
                                 'GCN', 'SAGE','MixHop', 'GATv2','GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--model_path',type=str, default='./model')
    parser.add_argument('--save_model',action='store_true', default=False)
    # model architecture
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model.')
    parser.add_argument('--hidden_dim', type=int, default=256, help=' modules input dimension.')
    parser.add_argument('--conv_layers', type=int, default=1, help='Number of conv layers in the GateMambaGCN.')
    parser.add_argument('--hidden_dim_multiplier', type=int, default=1, help='dim of edge encoder.')
    parser.add_argument('--d_state', type=int, default=2, help='mamba dimension of the hidden state or Number of expert')
    parser.add_argument('--expand', action='store_true', default=False, help='Expand the model.')
    parser.add_argument('--top_k', type=int, default=0.5, help='Number of topk expert.')
    parser.add_argument('--normalization', type=str, default='LayerNorm', help='Use Layer normalization.')
    parser.add_argument('--seed', type=str, default=0, help='Use batch normalization.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    # dataset
    parser.add_argument('--run_multiple_splits', type=list, default=[0,1,2,3,4,5,6,7,8,9],
                        help='[0,1,2,3,4,5,6,7,8,9]: ten times standard split. []: ten times seed of first split')
    parser.add_argument('--data_split_mode', type=str, default='standard',
                        choices=['standard', 'random', 'cv-'],
                        help='Mode for splitting the dataset into train/val/test splits.')
    parser.add_argument('--data_task', type=str, default='node', choices=['node', 'graph'],help='Task type for the dataset.')
    parser.add_argument('--data_split_index', type=int, default=0, help='Index of the dataset split to use.')
    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true') 
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    # regularization
    parser.add_argument('--dropout1', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout2', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--num_runs', type=int, default=1) 
    parser.add_argument('--patience', type=int, default=200, help='Patience for early stopping.')
    parser.add_argument('--num_warmup_steps', type=int, default=None, help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0.0, help='Only used if num_warmup_steps is None.')
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--dt_init', type=str, default='constant', choices=['random', 'constant'], help='Initialization method for time steps: random, constant')
    parser.add_argument('--pool', type=str, default='gcn', choices=['mean', 'max', 'min', 'sum', 'gcn'], help='Pooling method for node embeddings: mean, max, min, sum')
    
    # Optuna Settings
    parser.add_argument('--optruns', type=int, default=2000)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--Optuna_name', type=str, default="opt")
    
    # APPNP
    parser.add_argument('--alpha', type=int, default=0.7, help='Number of propagation steps.')
    
    args = parser.parse_args()
    if args.name is None:
        args.name = args.model

    return args

def run_loop_settings(args):
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(args.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.num_runs
        seeds = [args.seed + x for x in range(num_iterations)]
        split_indices = [args.data_split_index] * num_iterations
        run_ids = [x for x in range(num_iterations)]
        # seeds = [5025,5154,5034,5085, 1074,132, 139, 5177,190,5267]
        # seeds = [0,1,2,3,4,5,6,7,8,9]
        seeds=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363] # 别人 原
        # seeds=[3212139042,4023022221, 1941488137,4198936517,983997847,4019585660,2108550661,1648766618,629014539,2424918363] # 别人 
    else:
        # 'multi-split' run mode
        if args.num_runs != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(args.run_multiple_splits)
        split_indices = args.run_multiple_splits
        run_ids = split_indices
        seeds = [args.seed] * num_iterations
    return run_ids, seeds, split_indices
def compute_metrics(args, logits, dataset):
    if args.dataset in ['cora','pubmed','citeseer','computers','photo']:
        data = dataset[0]
    else:
        data = dataset._data
    if dataset.num_targets == 1:
        train_metric = roc_auc_score(y_true=data.y[data.train_mask].cpu().numpy(),
                                        y_score=logits[data.train_mask].cpu().numpy()).item()

        val_metric = roc_auc_score(y_true=data.y[data.val_mask].cpu().numpy(),
                                    y_score=logits[data.val_mask].cpu().numpy()).item()

        test_metric = roc_auc_score(y_true=data.y[data.test_mask].cpu().numpy(),
                                    y_score=logits[data.test_mask].cpu().numpy()).item()

    else:
        preds = logits.argmax(axis=1)
        train_metric = (preds[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_metric = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_metric = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    metrics = {
        f'train {dataset.metric}': train_metric,
        f'val {dataset.metric}': val_metric,
        f'test {dataset.metric}': test_metric
    }

    return metrics
def train_step(step, model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()
    if args.dataset in ['cora','pubmed','citeseer','computers','photo']:
        data = dataset[0]
    else:
        data = dataset._data
    with autocast(enabled=amp):
        logits = model(x=data.x, edge_index=data.edge_index, A_norm=data.norm_A, edge_attr=data.edge_attr)
        if dataset.num_targets > 1:
            loss = dataset.loss_fn(input=logits[data.train_mask], target=data.y[dataset.train_mask])
        else:
            loss = dataset.loss_fn(input=logits[data.train_mask], target=data.y[dataset.train_mask].unsqueeze(1).float())
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    optimizer.zero_grad()
    scheduler.step()

@torch.no_grad()
def evaluate(model, dataset, args):
    model.eval()
    if args.dataset in ['cora','pubmed','citeseer','computers','photo']:
        data = dataset[0]
    else:
        data = dataset._data
    with autocast(enabled=args.amp):
        logits = model(x=data.x, edge_index=data.edge_index,A_norm=data.norm_A, edge_attr=data.edge_attr)

    metrics = compute_metrics(args, logits, dataset)

    return metrics

def search_hyper_params(trial: optuna.Trial):
    
    # weight_decay = trial.suggest_categorical("weight_decay", [0.0005])
    # lr = trial.suggest_categorical("lr", [1e-4])
    # d_state = trial.suggest_categorical("d_state", [4])
    # dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    loop_num = trial.number
    num_layers = trial.suggest_categorical("num_layers", [16])
    dropout1 = trial.suggest_categorical("dropout1", [0, 0.5, 0.55, 0.6])
    dropout2 = trial.suggest_categorical("dropout2", [0, 0.1, 0.15, 0.2])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0001])
    lr = trial.suggest_categorical("lr", [0.001])
    d_state = trial.suggest_categorical("d_state", [2])
    hidden_dim = trial.suggest_categorical("hidden_dim", [256])
    return work(
                loop_num,
                hidden_dim,
                dropout1,
                dropout2,
                weight_decay,
                lr,
                d_state,
                num_layers
                )
def work(
        loop_num,
        hidden_dim,
        dropout1,
         dropout2,
         weight_decay,
         lr,
         d_state,
         num_layers
        ):
    args.hidden_dim = hidden_dim
    args.num_layers = num_layers
    args.dropout1 = dropout1
    args.dropout2 = dropout2
    args.weight_decay = weight_decay
    args.lr = lr
    args.d_state = d_state
    #args.num_warmup_steps = num_warmup_steps
    outs = []
    log_run = 0
    
        
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        args.split_index = split_index
        args.seed = seed
        # print('seed:',seed)
        seed_everything(args.seed)
        dataset = DataLoader(args).to(torch.device(args.device))
        
        seed_everything(0)
        if log_run == 0: 
            logger = Logger(args, metric=dataset.metric, num_data_splits=args.num_runs) 
        log_run = 1   
        
        if  args.model == 'APPNP':
            model = APPNPModule(
                        num_layers=args.num_layers,
                        input_dim=dataset.num_node_features,
                        hidden_dim=args.hidden_dim,
                        alpha=args.alpha,
                        output_dim=dataset.num_targets,
                        normalization=args.normalization,
                        dropout1=args.dropout1,
                        dropout2=args.dropout2)
        else:
            model = Model(model_name=args.model,
                            dt_init = args.dt_init,
                        num_layers=args.num_layers,
                        input_dim=dataset.num_node_features,
                        num_nodes=dataset.data.x.shape[0],
                        hidden_dim=args.hidden_dim,
                        conv_layes=args.conv_layers,
                        mamba_hidden_dim=args.d_state,
                        output_dim=dataset.num_targets,
                        hidden_dim_multiplier=args.hidden_dim_multiplier,
                        top_k=args.top_k,
                        num_heads=args.nb_heads,
                        normalization=args.normalization,
                        dropout1=args.dropout1,
                        dropout2=args.dropout2,
                        pool=args.pool)
        
        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer,                      # # 绑定的优化器（如Adam）
                                                 num_warmup_steps=args.num_warmup_steps,   # 预热步数
                                                 num_steps=args.num_steps,                 # 总训练步数
                                                 warmup_proportion=args.warmup_proportion) # # 预热比例（与num_warmup_steps二选一）

        logger.start_run(run=run_id + 1, data_split=split_index + 1, args=args)

        # with tqdm(total=args.num_steps, desc=f'Run {run_id}', disable=args.verbose) as progress_bar:
        for step in range(1, args.num_steps + 1):

            train_step(step, model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, amp=args.amp)
            metrics = evaluate(model=model, dataset=dataset, args=args)
                    
            logger.update_metrics(metrics=metrics, step=step)

                # progress_bar.update()
                # progress_bar.set_postfix({metric: f'{value:.4f}' for metric, value in metrics.items()})
                
        logger.finish_run(args)
        model.cpu()
        outs.append(logger.test_metrics[-1])
        # dataset.next_data_split()

    logger.print_metrics_summary(args)
    
    return np.average(outs)

def main(k):
    # # Load cmd line args
    # args = get_args()
    print(f'Running experiment {k}')
    log_run = 0
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        args.split_index = split_index
        args.seed = seed
        print('seed:',seed)
        seed_everything(args.seed)
        dataset = DataLoader(args).to(torch.device(args.device))
        seed_everything(0)
        if log_run == 0: 
            logger = Logger(args, metric=dataset.metric, num_data_splits=args.num_runs) 
        log_run = 1   
         
        if  args.model == 'APPNP':
            model = APPNPModule(
                        num_layers=args.num_layers,
                        input_dim=dataset.num_node_features,
                        hidden_dim=args.hidden_dim,
                        alpha=args.alpha, 
                        output_dim=dataset.num_targets,
                        normalization=args.normalization,
                        dropout1=args.dropout1,
                        dropout2=args.dropout2)
        else:
            model = Model(model_name=args.model,
                          dt_init = args.dt_init,
                        num_layers=args.num_layers,
                        input_dim=dataset.num_node_features,
                        num_nodes=dataset.data.x.shape[0],
                        hidden_dim=args.hidden_dim,
                        conv_layes=args.conv_layers,
                        mamba_hidden_dim=args.d_state,
                        output_dim=dataset.num_targets,
                        hidden_dim_multiplier=args.hidden_dim_multiplier,
                        num_heads=args.nb_heads,
                        top_k=args.top_k,
                        normalization=args.normalization,
                        dropout1=args.dropout1,
                        dropout2=args.dropout2,
                        pool=args.pool)
        
        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer,                      # # 绑定的优化器（如Adam）
                                                 num_warmup_steps=args.num_warmup_steps,   # 预热步数
                                                 num_steps=args.num_steps,                 # 总训练步数
                                                 warmup_proportion=args.warmup_proportion) # # 预热比例（与num_warmup_steps二选一）

        logger.start_run(run=run_id + 1, data_split=split_index + 1, args=args)

        val_score = 0
        early_stop = 0
        with tqdm(total=args.num_steps, desc=f'Run {run_id}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):

                train_step(step, model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, args=args)
                
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.4f}' for metric, value in metrics.items()})
                if metrics[f'val {dataset.metric}'] > val_score:
                    early_stop = 0
                    val_score =  metrics[f'val {dataset.metric}']
                    if args.save_model:
                        save_dir = f"saved_model/{args.dataset}"
                        os.makedirs(save_dir, exist_ok=True)
                        model_path = os.path.join(save_dir, f"{args.dataset}_{k}_{run_id}.pt")
                        torch.save(model.state_dict(), model_path)
                else:
                    early_stop += 1
                    
                if early_stop > args.patience:
                    break
                # logger.update_metrics(metrics=metrics, step=step)

                # progress_bar.update()
                # progress_bar.set_postfix({metric: f'{value:.4f}' for metric, value in metrics.items()})
                
        logger.finish_run(args)
        
        model.cpu()
        # dataset.next_data_split()

    logger.print_metrics_summary(args)



if __name__ == '__main__':
    # Load cmd line args
    
    args = get_args()
    
    if args.test:
        for i in range(10):
            main(i)
    else:
        study = optuna.create_study(direction="maximize",
                                    storage="sqlite:///" + args.path +
                                    args.dataset + ".db",
                                    study_name=args.Optuna_name,
                                    load_if_exists=True)
        study.optimize(search_hyper_params, n_trials=args.optruns)
        print("best params ", study.best_params)
        print("best valf1 ", study.best_value)        




