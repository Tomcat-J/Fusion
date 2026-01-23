import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from mdistiller.distillers import distiller_dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1, top5= [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))
    all_preds = []
    all_targets = []
    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            output = output.cuda(non_blocking=True)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)


            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    res,res2 = compute_dataset_metrics(val_loader,distiller)
    precision2, recall2, f2 = res2

    return top1.avg, top5.avg, losses.avg, precision2,recall2,f2,res

#通过不同的颜色来区分日志消息的重要性或类别
def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.cuda(non_blocking=True)
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")



def compute_metrics(predictions, targets, average='macro'):
    """
    计算并返回精确度、召回率、F1分数和准确度的平均值。
    参数:
    - predictions: 模型的预测输出
    - targets: 真实标签
    - average: 指定计算平均的方式，默认为'macro'平均
    """
    # 确保预测结果为类别标签形式
    with torch.no_grad():
        if predictions.dim() > 1:  # 如果predictions是one-hot或logits形式
            preds = predictions.argmax(dim=1)
        else:
            preds = predictions
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        # 计算精确度、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average=average, zero_division=1)

        # 计算准确度
        acc = accuracy_score(targets, preds)
        res = [acc*100, precision*100, recall*100, f1*100]

    return res


def compute_dataset_metrics(data_loader, distiller, device='cpu', average='macro'):
    """
    计算整个数据集的精确度、召回率、F1分数和准确度。

    参数:
    - data_loader: 提供数据的 DataLoader。
    - model: 用于预测的模型。
    - device: 设备类型，如 'cpu' 或 'cuda'。
    - average: 指定计算平均的方式，默认为 'macro' 平均。

    返回:
    - 一个包含整个数据集的准确度、精确度、召回率和 F1 分数的列表。
    """
    all_preds = []
    all_targets = []

    distiller.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 确保不计算梯度
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = distiller(image=data)
            if outputs.dim() > 1:
                preds = outputs.argmax(dim=1)
            else:
                preds = outputs

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算整个数据集的性能指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=1)
    res = [precision * 100, recall * 100, f1 * 100]
    precision2, recall2, f2, _ = precision_recall_fscore_support(all_targets, all_preds, average=average, zero_division=1)
    res2 = [precision2 * 100, recall2 * 100, f2 * 100]
    '''
    cm = confusion_matrix(all_preds, all_targets)
    classes = ['1ji', '2ji', '3ji', 'AIS', 'MIA']
    # 使用Seaborn绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    '''
    return res,res2