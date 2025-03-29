# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import os
import sys
from multiprocessing import Pool
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.special import softmax
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import utils

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.distributed as dist

import wandb

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    assert target.min() >= 0 and target.max() < 2
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    all_targets = []
    all_probs = []

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            # mixup handle 3th & 4th dimension
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
            samples = samples.view(B, C, T, H, W)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets,
                                             criterion)
        else:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, output = train_class_batch(model, samples, targets,
                                                 criterion)

        loss_value = loss.item()

        # Softmax to get probabilities (assuming binary classification)
        probs = F.softmax(output, dim=1)[:, 1]  # Take the prob of positive class

        # Collect for AUC
        all_probs.append(probs)
        all_targets.append(targets.view(-1))

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()

            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            acc1 = (output.max(-1)[-1] == targets).float().mean()
        else:
            acc1 = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(acc1=acc1)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(acc1=acc1, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    if dist.is_initialized():
        all_targets_list = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        all_probs_list = [torch.zeros_like(all_probs) for _ in range(dist.get_world_size())]
        dist.all_gather(all_targets_list, all_targets)
        dist.all_gather(all_probs_list, all_probs)

        all_targets = torch.cat(all_targets_list).cpu().numpy()
        all_probs = torch.cat(all_probs_list).cpu().numpy()

    # Calculate AUC
    auc = roc_auc_score(all_targets, all_probs) * 100

    # 计算TP，TN，FP，FN
    all_preds = (all_probs >= 0.5).astype(int)
    TP = int(np.sum((all_targets == 1) & (all_preds == 1)))
    TN = int(np.sum((all_targets == 0) & (all_preds == 0)))
    FP = int(np.sum((all_targets == 0) & (all_preds == 1)))
    FN = int(np.sum((all_targets == 1) & (all_preds == 0)))

    if log_writer is not None:
        log_writer.update(train_AUC=auc, head="loss")

    print("Averaged stats:", metric_logger, "train_AUC:", auc)

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['AUC'] = auc
    results['TP'] = TP
    results['TN'] = TN
    results['FP'] = FP
    results['FN'] = FN

    return results


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    all_targets = []
    all_probs = []

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        # Softmax to get probabilities (assuming binary classification)
        probs = F.softmax(output, dim=1)[:, 1]  # Take the prob of positive class

        # Collect for AUC
        all_probs.append(probs)
        all_targets.append(target.view(-1))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    if dist.is_initialized():
        all_targets_list = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        all_probs_list = [torch.zeros_like(all_probs) for _ in range(dist.get_world_size())]
        dist.all_gather(all_targets_list, all_targets)
        dist.all_gather(all_probs_list, all_probs)

        all_targets = torch.cat(all_targets_list).cpu().numpy()
        all_probs = torch.cat(all_probs_list).cpu().numpy()

    # Calculate AUC
    auc = roc_auc_score(all_targets, all_probs) * 100

    # 计算TP，TN，FP，FN
    all_preds = (all_probs >= 0.5).astype(int)
    TP = int(np.sum((all_targets == 1) & (all_preds == 1)))
    TN = int(np.sum((all_targets == 0) & (all_preds == 0)))
    FP = int(np.sum((all_targets == 0) & (all_preds == 1)))
    FN = int(np.sum((all_targets == 1) & (all_preds == 0)))

    print('* AUC {auc:.3f} Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(auc=auc, top1=metric_logger.acc1, losses=metric_logger.loss))

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['AUC'] = auc
    results['TP'] = TP
    results['TN'] = TN
    results['FP'] = FP
    results['FN'] = FN

    return results


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Final Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    all_targets = []
    all_probs = []

    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())),
                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        # Softmax to get probabilities (assuming binary classification)
        probs = F.softmax(output, dim=1)[:, 1]  # Take the prob of positive class

        # Collect for AUC
        all_probs.append(probs)
        all_targets.append(target.view(-1))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}\n".format(acc1))
        for line in final_result:
            f.write(line)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if target.dim() == 0:
        target = target.unsqueeze(0)

    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    if dist.is_initialized():
        all_targets_list = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        all_probs_list = [torch.zeros_like(all_probs) for _ in range(dist.get_world_size())]
        dist.all_gather(all_targets_list, all_targets)
        dist.all_gather(all_probs_list, all_probs)

        all_targets = torch.cat(all_targets_list).cpu().numpy()
        all_probs = torch.cat(all_probs_list).cpu().numpy()

    # Calculate AUC
    auc = roc_auc_score(all_targets, all_probs) * 100

    print('* AUC {auc:.3f} Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(auc=auc, top1=metric_logger.acc1, losses=metric_logger.loss))

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['AUC'] = auc

    return results


def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            if method == 'prob':
                dict_feats[name].append(softmax(data))
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, label, prob]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    label = [x[2] for x in ans]
    prob = [x[3] for x in ans]

    final_top1= np.mean(top1)
    final_auc = roc_auc_score(label, prob)

    return final_top1 * 100, final_auc * 100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    prob = feat[1]
    return [pred, top1, label, prob]
