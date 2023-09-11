import torch
from torch_geometric.data import DataLoader
from gnn_no_learn import GNN

import argparse
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import datetime
import statistics

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

import sys
import optuna
import joblib
import wandb

sys.path.insert(0, '../..')
from attacks import *

parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--gnn', type=str, default='gcn',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv, ogbg-molpcba)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--step-size', type=float, default=1e-3)
parser.add_argument('--step_m', type=int, default=3)
parser.add_argument('-m', type=int, default=3)
parser.add_argument('--test-freq', type=int, default=1)
parser.add_argument('--attack', type=str, default='flag')
parser.add_argument('--flag', type=int, default=0)
parser.add_argument('--order', type=int, default=1)
parser.add_argument('--sample', type=int, default=1)

parser.add_argument('--grad_clip', type=float, default=0.,
                    help='gradient clipping')
parser.add_argument('--warmup', type=float,
                    default=1000, help='consistency loss warmup')
args = parser.parse_args()
print("args: ", args)

def train_vanilla(epoch, num_batch, model, device, loader, train_test_loader, optimizer, task_type, grad_clip, order):
    model.train()
    loss_list = []
    order, sample = args.order, args.sample
    for step, batch in enumerate(loader):
        test_pred_list = []
        batch = batch.to(device)
        test_data = next(iter(train_test_loader)).to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            y = batch.y.to(torch.float32)[is_labeled]
            train_pred = model(batch).to(torch.float32)[is_labeled]
            loss_train = cls_criterion(train_pred, y)
            alpha, lam = 1, 1
            for k in range(sample):
                test_pred = model(test_data).to(torch.float32)
                test_pred_list.append(test_pred)
            loss_consis = consis_loss(test_pred_list)
            lam = min(lam, (lam * float(num_batch) / args.warmup))
            # lam = min(lam, epoch * 0.1)
            loss = alpha * loss_train + lam * loss_consis
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    grad_clip)
            optimizer.step()
            loss_list.append(loss.item())
            num_batch += 1

    return statistics.mean(loss_list), num_batch


def train(epoch, num_batch, model, device, train_loader, train_test_loader, optimizer, task_type, args):
    total_loss = 0
    sample, order = args.sample, args.order
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        test_data = next(iter(train_test_loader)).to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

            forward = lambda perturb: model(batch, perturb).to(torch.float32)[is_labeled]
            y = batch.y.to(torch.float32)[is_labeled]
            train_perturb_shape = (batch.x.shape[0], 300)
            test_perturb_shape = (test_data.x.shape[0], 300)
            perturb_shape = (train_perturb_shape, test_perturb_shape)
            test_forward = lambda perturb: model(test_data, perturb).to(torch.float32)
            model_forward = (model, forward, test_forward)
            loss, _ = flag_cr(epoch, num_batch, model_forward, perturb_shape, y, args, optimizer, device, cls_criterion)
            total_loss += loss.item()
        num_batch += 1
    return total_loss / len(train_loader), num_batch


def eval(model, device, loader, evaluator, order):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def opt_objective(trial, args):
    valid_res_list = []
    test_res_list = []
    args.conf = trial.suggest_float('conf', 0.0, 1)
    args.sample = trial.suggest_int('sample', 1, 5)
    for run in range(args.runs):
        # args.run = i + 1
        for seed in range(1):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(args)
            valid_result, test_result = main(args, run)
            valid_res_list.append(valid_result)
            test_res_list.append(test_result)
    print(valid_res_list)
    print(test_res_list)
    print(f'Obj valid avg accuracy: {np.mean(valid_res_list)}, {np.std(valid_res_list)}')
    print(f'Obj test avg accuracy: {np.mean(test_res_list)}, {np.std(test_res_list)}')
    return np.mean(valid_res_list)


def run_best(args, study):
    args.conf = study.best_trial.params['conf']
    args.sample = study.best_trial.params['sample']
    # args.order = study.best_trial.params['order']
    print("args: ", args)
    valid_res_list = []
    test_res_list = []
    for run in range(args.runs):
        for seed in range(1):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            valid_result, test_result = main(args, run)
            valid_res_list.append(valid_result)
            test_res_list.append(test_result)
    print(valid_res_list)
    print(test_res_list)

    print('')
    print(f"Best Average val accuracy: {np.mean(valid_res_list)} ± {np.std(valid_res_list)}")
    print(f"Best Average test accuracy: {np.mean(test_res_list)} ± {np.std(test_res_list)}")
    print(study.best_trial.params)


def main(args, run):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]
    print("args: ", args)
    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    train_test_loader = test_loader
    vals, tests = [], []
    s_t = datetime.datetime.now()
    best_val, final_test = 0, 0
    final_epoch = 0
    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , weight_decay=5e-4

    wandb_log = {}
    num_batch = 0
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            args.grad_clip)
    for epoch in range(1, args.epochs + 1):
        if args.flag:
            loss, num_batch_ = train(epoch, num_batch, model, device, train_loader, train_test_loader, optimizer,
                                     dataset.task_type, args)
        else:
            loss, num_batch_ = train_vanilla(epoch, num_batch, model, device, train_loader, train_test_loader,
                                             optimizer,
                                             dataset.task_type, args.grad_clip, args)
        num_batch += num_batch_
        if epoch > args.epochs // 100 and epoch % args.test_freq == 0 or epoch == args.epochs:
            train_perf = eval(model, device, train_loader, evaluator, args.order)
            valid_perf = eval(model, device, valid_loader, evaluator, args.order)
            test_perf = eval(model, device, test_loader, evaluator, args.order)

            print({'Epoch': epoch, 'Train': train_perf['rocauc'], 'Validation': valid_perf['rocauc'],
                   'Test': test_perf['rocauc'], 'Loss': loss, 'Num_batch': num_batch})
            result = (
                train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric])
            _, val, tst = result
            if val > best_val:
                best_val = val
                final_test = tst
                final_epoch = epoch
            wandb_log['Epoch'] = epoch
            wandb_log['seed'] = run
            wandb_log['loss_train'] = loss
            wandb_log['train_' + dataset.eval_metric] = train_perf[dataset.eval_metric]
            wandb_log['valid_' + dataset.eval_metric] = valid_perf[dataset.eval_metric]
            wandb_log['test_' + dataset.eval_metric] = test_perf[dataset.eval_metric]
            wandb.log(wandb_log)
    print(f'Run{run} val:{best_val}, test:{final_test}, epoch:{final_epoch}')
    e_t = datetime.datetime.now()
    print(f'start time:{s_t} end time:{e_t}')
    vals.append(best_val)
    tests.append(final_test)
    print("Seed:", run)
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")
    return best_val, final_test


if __name__ == "__main__":
    s = datetime.datetime.now()
    print("time of start: ", s)
    search_space = {'conf': [0.2, 0.5, 0.8, 0.9], 'sample': [1, 2, 3, 4]}  # , 'order': [1, 2, 3, 4]}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: opt_objective(trial, args), n_trials=12)
    joblib.dump(study, f'{args.gnn}_{args.dataset}_{args.conf}_{args.sample}_{args.order}_{args.flag}_study_cr.pkl')
    run_best(args, study)
    e = datetime.datetime.now()
    print("time of end: ", e)
