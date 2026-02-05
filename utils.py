import os
import yaml
import numpy as np
import torch
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
class Logger:
    def __init__(self, args, metric, num_data_splits):
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.dataset = args.dataset
        self.verbose = args.verbose
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.num_runs = args.num_steps
        self.num_data_splits = num_data_splits
        self.cur_run = None
        self.cur_data_split = None
        if args.test:
            print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run, data_split, args):
        self.cur_run = run
        self.cur_data_split = data_split
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.best_steps.append(None)
        if args.test:
            if self.num_data_splits == 1:
                print(f'Starting run {run}/{10}...')
            else:
                print(f'Starting run {run}/{10} (using data split {data_split}/{self.num_data_splits})...')

    def update_metrics(self, metrics, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.4f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.4f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.4f}')

    def finish_run(self,args):
        self.save_metrics()
        if args.test:
            print(f'Finished run {self.cur_run}. '
                f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
                f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
                f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        if self.dataset in ['cora','pubmed','citeseer']:
            val_metric_mean = np.mean(self.val_metrics).item()
            val_metric_std = np.std(self.val_metrics).item() if len(self.val_metrics) > 1 else np.nan
            test_metric_mean = np.mean(self.test_metrics).item()
            test_metric_std = np.std(self.test_metrics).item() if len(self.test_metrics) > 1 else np.nan
        else:
            val_metric_mean = np.mean(self.val_metrics).item()
            val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
            test_metric_mean = np.mean(self.test_metrics).item()
            test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self,args):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)
        
        if args.test:
            print(f'Finished {metrics["num runs"]} runs.')
            print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
            print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
            print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
            print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')
    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask
def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    # train = index_to_index(train_idx, size=data.num_nodes)

    return data


def get_parameter_groups1(model):
    # 无权重衰减的参数名关键词
    no_weight_decay_names = ['bias', 'label_embeddings']
    
    # 需要特殊权重衰减的参数名关键词
    special_weight_decay_names = ['A_log', 'x_proj', 'dt_proj']
    
    # 收集不同组的参数
    special_params = []  # 特殊权重衰减 (0.01)
    no_decay_params = []  # 无权重衰减 (0)
    normal_params = []    # 正常权重衰减 (默认)
    
    # 遍历所有参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 检查是否是无权重衰减参数
        if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names):
            no_decay_params.append(param)
        # 检查是否是特殊权重衰减参数
        elif any(special_name in name for special_name in special_weight_decay_names):
            special_params.append(param)
        # 其他参数使用默认权重衰减
        else:
            normal_params.append(param)
    
    # 创建参数组
    parameter_groups = []
    
    # 正常参数组（使用默认权重衰减）
    if normal_params:
        parameter_groups.append({
            'params': normal_params,
            'weight_decay': 0.000001 # 可以在此设置默认值，或在优化器中统一设置
        })
    
    # 特殊权重衰减参数组
    if special_params:
        parameter_groups.append({
            'params': special_params,
            'weight_decay': 0.01  # 特殊权重衰减值
        })
    
    # 无权重衰减参数组
    if no_decay_params:
        parameter_groups.append({
            'params': no_decay_params,
            'weight_decay': 0.0  # 无权重衰减
        })

    


    return parameter_groups
def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']

    
    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
