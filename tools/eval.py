import argparse
import importlib
import os
import subprocess
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from mae_lite.utils import (
    DictAction,
    AvgMeter,
    accuracy,
    setup_logger,
    collect_env_info,
    random_seed,
    remove_possible_prefix,
)
from mae_lite.utils.torch_dist import parse_devices, configure_nccl, all_reduce_mean, synchronize
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from mae_lite.exps import timm_imagenet_exp
from projects.eval_tools import finetuning_rpe_exp, finetuning_exp
from projects.mae_lite.mae_lite_distill_exp import set_model_weights

def get_arg_parser():
    parser = argparse.ArgumentParser("Classification Evaluation")
    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("-d", "--devices", default="0", type=str, help="device for training")
    parser.add_argument(
        "-f",
        "--exp_file",
        # default=timm_imagenet_exp.__file__,
        default=finetuning_exp.__file__,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--ckpt", default="/home/backup/lh/KD/outputs/HiFuse_Small_1e-5-0.05/ft_eval/last_epoch_best_ckpt.checkpoints.tar", type=str, help="checkpoint path for evaluation")
    parser.add_argument(
        "--exp-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
        'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_arg_parser()
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = len(args.devices.split(","))

    nr_machine = int(os.getenv("MACHINE_TOTAL", "1"))
    if nr_gpu > 1:
        args.world_size = nr_gpu * nr_machine
        processes = []
        for rank in range(nr_gpu):
            p = mp.Process(target=main_worker, args=(rank, nr_gpu, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        main_worker(0, nr_gpu, args)


def main_worker(gpu, nr_gpu, args):
    current_exp_name = os.path.basename(args.exp_file).split(".")[0]
    # ------------ set environment variables for distributed training ------------------------------------- #
    configure_nccl()
    rank = gpu
    if nr_gpu > 1:
        rank += int(os.getenv("MACHINE_RANK", "0")) * nr_gpu

        if args.dist_url is None:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            master_ip = str(master_ip).strip()
            args.dist_url = "tcp://{}:23456".format(master_ip)

            # ------------------------hack for multi-machine training -------------------- #
            if args.world_size > 8:
                ip_add_file = "./" + current_exp_name + "ip_add.txt"
                if rank == 0:
                    with open(ip_add_file, "w") as ip_add:
                        ip_add.write(args.dist_url)
                else:
                    while not os.path.exists(ip_add_file):
                        time.sleep(0.5)

                    with open(ip_add_file, "r") as ip_add:
                        dist_url = ip_add.readline()
                    args.dist_url = dist_url
        else:
            args.dist_url = "tcp://{}:23456".format(args.dist_url)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )
        print("Rank {} initialization finished.".format(rank))
        synchronize()

        if rank == 0:
            if os.path.exists("./" + current_exp_name + "ip_add.txt"):
                os.remove("./" + current_exp_name + "ip_add.txt")

    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])

    exp = current_exp.Exp(args.batch_size)
    update_cfg_msg = exp.update(args.exp_options)

    if exp.seed is not None:
        random_seed(exp.seed, rank)

    file_name = os.path.join(exp.output_dir, exp.exp_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    logger = setup_logger(file_name, distributed_rank=rank, filename="eval_log.txt", mode="a")
    if rank == 0:
        logger.info("gpuid: {}, args: <>{}".format(rank, args))
        logger.opt(ansi=True).info(
            "<yellow>Used experiment configs</yellow>:\n<blue>{}</blue>".format(exp.get_cfg_as_str())
        )
        if update_cfg_msg:
            logger.opt(ansi=True).info(
                "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
            )
        logger.opt(ansi=True).info("<yellow>Environment info:</yellow>\n<blue>{}</blue>".format(collect_env_info()))

    data_loader = exp.get_data_loader()
    eval_loader = data_loader["eval"]
    model = exp.get_model()
    if rank == 0:
        logger.info("Illustration of model strcutures:\n{}".format(str(model)))
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if nr_gpu > 1:
        ddp_model = DDP(model, device_ids=[gpu])
    else:
        ddp_model = model

    #  ------------------------------------------- load ckpt ------------------------------------ #
    # specify the path of the ckeckpoint for evaluation.
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_path = args.ckpt 
    else:
        # Automaticly load the lastest checkpoint.
        ckpt_path = os.path.join(file_name, "last_epoch_best_ckpt.checkpoints.tar")
        if os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(file_name, "last_epoch_ckpt.checkpoints.tar")
            assert os.path.isfile(ckpt_path), "Failed to load ckpt from '{}'".format(ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    student_weights_prefix = ""
    msg = set_model_weights(model,ckpt_path,student_weights_prefix)
    # msg = model.load_state_dict(remove_possible_prefix(ckpt["model"]))
    if rank == 0:
        logger.warning("Model params {} are not loaded".format(msg.missing_keys))
        logger.warning("State-dict params {} are not used".format(msg.unexpected_keys))

    ddp_model.eval()
    eval_top1, eval_top3 ,precision, recall, f1,_,_= run_eval(ddp_model, eval_loader)
    if rank == 0:
        logger.info("Evaluation of experiment: {} is done.".format(exp.exp_name))
        logger.info(
            "\tTop1:{:.3f}, Top3:{:.3f},".format(eval_top1, eval_top3)
        )
    logger.stop()

def run_eval(model, eval_loader):
    """
    评估函数 - 适配新的多原型分类头
    
    模型返回格式: (logits, head_loss)
    - logits: 分类 logits
    - head_loss: Head 内部计算的损失 (CE + proto_loss)
    """
    top1 = AvgMeter()
    top3 = AvgMeter()
    losses = AvgMeter()
    all_targets = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluating")
        for inp, target in pbar:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            # 模型返回 (logits, head_loss)
            logits, head_loss = model(inp, target)
            _, preds = torch.max(logits, 1)
            
            # 使用 head_loss 作为主要损失 (已包含 CE + proto_loss)
            # 额外计算一个纯 CE loss 用于监控
            ce_loss = criterion(logits, target)
            total_loss = head_loss if head_loss.item() > 0 else ce_loss
            
            acc1, acc3 = accuracy(logits, target, (1, 3))
            acc1, acc3 = all_reduce_mean(acc1), all_reduce_mean(acc3)
            losses.update(total_loss.cpu().detach().numpy().mean(), inp.size(0))
            top1.update(acc1.item(), inp.size(0))
            top3.update(acc3.item(), inp.size(0))

            # 保存预测和目标，以便后续计算Precision，Recall，F1
            all_predictions.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    # 计算Precision, Recall, F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=1)
    # 自动推断类别数
    num_classes = max(max(all_targets), max(all_predictions)) + 1
    specificity = calculate_specificity(all_targets, all_predictions, num_classes=num_classes)
    # # 计算混淆矩阵
    # conf_matrix = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2])
    # tn = conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[2, 2] - conf_matrix.trace()
    # fp = conf_matrix.sum(axis=0) - conf_matrix.trace()
    # specificity = tn / (tn + fp) if (tn + fp).sum() > 0 else 0

    print(f'acc1: {top1.avg:.4f}, acc3: {top3.avg:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Specificity: {specificity:.4f}')

    return top1.avg, top3.avg, precision, recall, f1,specificity,losses.avg
# def run_eval(model, eval_loader):
#     batch_time, losses, top1, top5= [AvgMeter() for _ in range(4)]
#     criterion = nn.CrossEntropyLoss()
#     num_iter = len(eval_loader)
#     pbar = tqdm(range(num_iter))
#     all_preds = []
#     all_targets = []
#     model.eval()
#     with torch.no_grad():
#         start_time = time.time()
#         for idx, (image, target) in enumerate(eval_loader):
#             image = image.float()
#             image = image.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#             output = model(image)
#             output = output.cuda(non_blocking=True)
#             loss = criterion(output, target)
#             acc1, acc5 = accuracy(output, target, topk=(1, 3))
#             batch_size = image.size(0)
#             losses.update(loss.cpu().detach().numpy().mean(), batch_size)
#             top1.update(acc1.item(), batch_size)
#             top5.update(acc1.item(), batch_size)
#
#
#     #         # measure elapsed time
#     #         batch_time.update(time.time() - start_time)
#     #         start_time = time.time()
#     #         msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
#     #             top1=top1, top5=top5
#     #         )
#     #         pbar.set_description(log_msg(msg, "EVAL"))
#     #         pbar.update()
#     # pbar.close()
#     res,res2 = compute_dataset_metrics(eval_loader,model)
#     precision2, recall2, f2 = res2
#
#     return top1.avg, top5.avg, losses.avg, precision2,recall2,f2,res


# #通过不同的颜色来区分日志消息的重要性或类别
# def log_msg(msg, mode="INFO"):
#     color_map = {
#         "INFO": 36,
#         "TRAIN": 32,
#         "EVAL": 31,
#     }
#     msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
#     return msg


def calculate_specificity(y_true, y_pred, num_classes):

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_list = []
    # classes = ['G1G2',  'G3', 'AISMIA']
    # # 使用Seaborn绘制混淆矩阵
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.show()

    for i in range(num_classes):
        true_negative = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        false_positive = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0
        specificity_list.append(specificity)

    # 返回所有类别的特异性的平均值
    return np.mean(specificity_list)

def compute_dataset_metrics(data_loader, distiller, device='cuda', average='macro'):
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
            outputs = distiller(data)
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

if __name__ == "__main__":
    main()
