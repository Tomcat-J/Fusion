# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import argparse
import importlib
import os
import subprocess
import sys
import time
import torch
import json

import math
from contextlib import suppress
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from mae_lite.tools.eval import run_eval
from mae_lite.utils import (
    TORCH_VERSION,
    DictAction,
    AvgMeter,
    DataPrefetcher,
    save_checkpoint,
    setup_logger,
    setup_tensorboard_logger,
    collect_env_info,
    Scaler,
    NativeScaler,
    random_seed,
    find_free_port
)
from mae_lite.utils.torch_dist import parse_devices, configure_nccl, all_reduce_mean, synchronize
from mae_lite.exps import timm_imagenet_exp
from projects.mae_lite import mae_lite_distill_exp
from projects.eval_tools import finetuning_exp, finetuning_transfer_exp
from projects.mae_lite.temp_global import Global_T


def get_arg_parser():
    parser = argparse.ArgumentParser("Training")
    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str,
                        help="url used to set up distributed training, e.g. 'tcp://127.0.0.1:8686'.")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("-e", "--max_epoch", type=int, default=100, help="max_epoch for training")
    parser.add_argument("-d", "--devices", default="0", type=str, help="device for training")
    parser.add_argument(
        "-eval",
        "--eval",
        action="store_true",
        help="enable evaluation during training, notice that self-supervised model are not supported evaluation",
    )
    parser.add_argument("--no_eval", action="store_false", dest="eval")
    parser.set_defaults(eval=True)
    parser.add_argument(
        "-f",
        "--exp_file",
        # default=timm_imagenet_exp.__file__,
        # default=mae_lite_distill_exp.__file__,
        default= finetuning_exp.__file__,
        # default=finetuning_transfer_exp.__file__,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--resume", dest="resume", nargs="?", default=argparse.SUPPRESS, type=str, help="path to latest checkpoint"
    )
    parser.add_argument("--ckpt", default="/home/backup/lh/KD/projects/mae_lite/checkpoints/HiFuse_ImageNet1K.pth",
                        type=str, help="checkpoint for initialization")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    parser.add_argument(
        "--exp-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
             'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             "Note that the quotation marks are necessary and that no white space is allowed.",
    )

    # CTKD distillation
    # parser.add_argument('--have_mlp', type=int, default=1)
    # parser.add_argument('--mlp_name', type=str, default='global')
    # parser.add_argument('--cosine_decay', type=int, default=1)
    # parser.add_argument('--decay_max', type=float, default=0)
    # parser.add_argument('--decay_min', type=float, default=-1)
    # parser.add_argument('--decay_loops', type=float, default=10)

    args = parser.parse_args()
    return args


class CosineDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                 max_value,
                 min_value,
                 num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value


def main():
    args = get_arg_parser()
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = len(args.devices.split(","))
    nr_machine = int(os.getenv("MACHINE_TOTAL", "1"))
    args.world_size = nr_gpu * nr_machine

    configure_nccl()
    if args.world_size > 1:
        if args.dist_url is None:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            port = find_free_port()
            master_ip = str(master_ip).strip()
            args.dist_url = "tcp://{}:{}".format(master_ip, port)
            # print(args.dist_url)

            # ------------------------hack for multi-machine training -------------------- #
            if nr_machine > 1:
                current_exp_name = os.path.basename(args.exp_file).split(".")[0]
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
    rank = gpu
    if args.world_size > 1:
        rank += int(os.getenv("MACHINE_RANK", "0")) * nr_gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )
        synchronize()

        if rank == 0:
            current_exp_name = os.path.basename(args.exp_file).split(".")[0]
            if os.path.exists("./" + current_exp_name + "ip_add.txt"):
                os.remove("./" + current_exp_name + "ip_add.txt")
        print("Rank {} initialization finished.".format(rank))

    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])

    if args.max_epoch:
        exp = current_exp.Exp(batch_size=args.batch_size, max_epoch=args.max_epoch)
    else:
        exp = current_exp.Exp(batch_size=args.batch_size)
    update_cfg_msg = exp.update(args.exp_options)

    if exp.seed is not None:
        random_seed(exp.seed, rank)

    file_name = os.path.join(exp.output_dir, exp.exp_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    logger = setup_logger(file_name, distributed_rank=rank, filename="train_log.txt", mode="a")

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

    if exp.enable_tensorboard:
        tb_writer = setup_tensorboard_logger(file_name, distributed_rank=rank, name="tb_train")
    else:
        tb_writer = None

    data_loader = exp.get_data_loader()
    train_loader = data_loader["train"]
    eval_loader = data_loader.get("eval", None)
    active_eval = args.eval and eval_loader is not None
    model = exp.get_model()
    # temp_mlp = None
    #
    # if args.have_mlp:
    #     if args.mlp_name == 'global':
    #         temp_mlp = Global_T()
    #     else:
    #         print('temp_mlp name wrong')
    # if args.cosine_decay:
    #     gradient_decay = CosineDecay(max_value=args.decay_max, min_value=args.decay_min, num_loops=args.decay_loops)
    # else:
    #     gradient_decay = LinearDecay(max_value=args.decay_max, min_value=args.decay_min, num_loops=args.decay_loops)
    #
    # decay_value = 1

    if rank == 0:
        logger.info("Illustration of model strcutures:\n{}".format(str(model)))
    optimizer = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if nr_gpu > 1:
        model = DDP(model, device_ids=[gpu])

    # ------------------------ start training ------------------------------------------------------------ #
    ITERS_PER_EPOCH = len(train_loader)
    BEST_TOP1_ACC = 0.0
    BEST_TOP3_ACC = 0.0
    BEST_TOP1_EPOCH = 0
    start_epoch = 0

    if args.amp:
        assert TORCH_VERSION >= (1, 6), "Automatic Mixed Precision is not supported for current pytorch version!"
        logger.info("Automatic Mixed Precision is enabled!")
        autocast = torch.cuda.amp.autocast
        scaler = NativeScaler()
    else:
        autocast = suppress
        scaler = Scaler()

    #  ------------------------------------------- load ckpt ------------------------------------ #
    if "resume" in args:
        if args.resume is None:
            args.resume = "last_epoch_ckpt.pth.tar"
            resume_path = os.path.join(file_name, args.resume)
        if os.path.isfile(args.resume):
            resume_path = args.resume
        else:
            resume_path = os.path.join(file_name, args.resume)
        if os.path.isfile(resume_path):
            logger.info("\tloading checkpoint '{}'".format(resume_path))
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(gpu)
            ckpt = torch.load(resume_path, map_location=loc)
            # resume the training states variables
            start_epoch = ckpt["start_epoch"]
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt.get("scaler", {}))
            scheduler.load_state_dict(ckpt.get("scheduler", {}))
            BEST_TOP1_ACC = ckpt.get("best_top1", 0.0)
            BEST_TOP3_ACC = ckpt.get("best_top3", 0.0)
            BEST_TOP1_EPOCH = ckpt.get("best_top1_epoch", 0)
            logger.info(
                "\tloaded successfully (epoch={}, BEST_TOP1_ACC={}(epoch={}))".format(
                    start_epoch, BEST_TOP1_ACC, BEST_TOP1_EPOCH
                )
            )
        else:
            raise FileNotFoundError("\tno checkpoint found at '{}'".format(args.resume))
    exp.set_current_state(start_epoch * ITERS_PER_EPOCH, ckpt_path=args.ckpt)
    #  ---------------------------------- train ------------------------------ #
    if rank == 0:
        logger.info("Training start...")

    scheduler.step(start_epoch * ITERS_PER_EPOCH)

    # ----------------------------------------- Backbone Freezing 设置 --------------------------------- #
    freeze_backbone_epochs = getattr(exp, 'freeze_backbone_epochs', 0)
    backbone_frozen = False
    
    # 获取内部模型 (穿透 DDP 和 Model wrapper)
    if hasattr(model, 'module'):
        wrapper_model = model.module  # DDP wrapper
    else:
        wrapper_model = model
    
    if hasattr(wrapper_model, 'model'):
        inner_model = wrapper_model.model  # Model wrapper 内部的 main_model
    else:
        inner_model = wrapper_model
    
    # 如果设置了 freeze_backbone_epochs 且从头开始训练，冻结 backbone
    if freeze_backbone_epochs > 0 and start_epoch < freeze_backbone_epochs:
        if hasattr(inner_model, 'freeze_backbone'):
            inner_model.freeze_backbone()
            backbone_frozen = True
            if rank == 0:
                logger.info(f"Backbone frozen for first {freeze_backbone_epochs} epochs (Head warmup)")
        else:
            if rank == 0:
                logger.warning("Model does not have freeze_backbone method, skipping backbone freezing")

    prefetcher = DataPrefetcher(train_loader, data_format=exp.data_format)
    for epoch in range(start_epoch, exp.max_epoch):
        batch_time_meter = AvgMeter()
        if rank == 0:
            logger.info("---> start train epoch{}".format(epoch + 1))

        # ----------------------------------------- Backbone 解冻检查 --------------------------------- #
        # 在 freeze_backbone_epochs 结束后解冻 backbone
        if backbone_frozen and epoch >= freeze_backbone_epochs:
            if hasattr(inner_model, 'unfreeze_backbone'):
                inner_model.unfreeze_backbone()
                backbone_frozen = False
                if rank == 0:
                    logger.info(f"Backbone unfrozen at epoch {epoch + 1} (Head warmup completed)")
            else:
                if rank == 0:
                    logger.warning("Model does not have unfreeze_backbone method")

        if prefetcher.next_input is None:
            if nr_gpu > 1:
                train_loader.sampler.set_epoch(epoch)
            prefetcher = DataPrefetcher(train_loader, data_format=exp.data_format)

        model.train()
        for i in range(ITERS_PER_EPOCH):
            iter_count = epoch * ITERS_PER_EPOCH + i + 1
            iter_start_time = time.time()
            inps, target = prefetcher.next()

            data_time = time.time() - iter_start_time

            optimizer.zero_grad()
            # if args.have_mlp:
            #     decay_value = gradient_decay.get_value(epoch)
            with autocast():
            # losses_dict, extra_dict, temp = model(inps, target=target, epoch=epoch, temp_mlp=temp_mlp,
                                                      # cos_value=decay_value)
                loss, extra_dict = model(inps, target=target, epoch=epoch)
            # backward
            # loss = sum([l.mean() for l in losses_dict.values()])
            loss_value = all_reduce_mean(loss).item()
            extra_dict = {k: all_reduce_mean(v) for k, v in extra_dict.items()} if extra_dict else None
            if not math.isfinite(loss_value):
                logger.warning("Loss is {:.4f}, Stop training".format(loss_value))
                sys.exit(1)
            scaler(loss, optimizer, clip_grad=exp.clip_grad, clip_mode=exp.clip_mode, parameters=model.parameters())

            # lr = update_lr_func(iter_count)
            scheduler.step(iter_count)

            batch_time_meter.update(time.time() - iter_start_time)
            if rank == 0 and (i + 1) % exp.print_interval == 0:
                remain_time = (ITERS_PER_EPOCH * exp.max_epoch - iter_count) * batch_time_meter.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)
                remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))

                lr_str = scheduler.get_last_lr_str()

                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                log_str = (
                    "[{}/{}], remain:{}, It:[{}/{}], Max-Mem:{:.0f}M, Data-Time:{:.3f}, LR:{}, Train_Loss:{:.4f}".format(
                        epoch + 1,
                        exp.max_epoch,
                        remain_time,
                        i + 1,
                        ITERS_PER_EPOCH,
                        max_mem_mb,
                        data_time,
                        lr_str,
                        loss_value,
                    )
                )
                if extra_dict:
                    extra_str = ", ".join(["{}:{}".format(k, v) for k, v in extra_dict.items()])
                    log_str = "{}, {}".format(log_str, extra_str)
                logger.info(log_str)

                if tb_writer:
                    tb_writer.add_scalar("Loss", loss_value, iter_count)
                    if extra_dict:
                        for k, v in extra_dict.items():
                            tb_writer.add_scalar(k, v, iter_count)

        # ----------------------------------------- push prototypes --------------------------------- #
        # 优化的原型推送策略 (避免 cooldown 阶段 push)
        push_interval = getattr(exp, 'push_interval', 15)  # 默认每15个epoch推送一次
        push_start_epoch = getattr(exp, 'push_start_epoch', 20)  # 默认从第20个epoch开始push
        push_end_epoch = getattr(exp, 'push_end_epoch', exp.max_epoch - 50)  # 默认在最后50个epoch前停止push
        push_momentum = getattr(exp, 'push_momentum', 0.95)  # 高动量=渐进式更新
        
        # 获取当前学习率，判断是否在 cooldown 阶段
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.lr
        min_lr_threshold = getattr(exp, 'min_lr', 1e-6) * 10  # LR 低于此值视为 cooldown
        is_cooldown = current_lr < min_lr_threshold
        
        # 获取内部模型 (需要穿透 DDP 和 Model wrapper)
        if hasattr(model, 'module'):
            wrapper_model = model.module  # DDP wrapper
        else:
            wrapper_model = model
        
        # 穿透 Model wrapper 获取实际的 main_model
        if hasattr(wrapper_model, 'model'):
            inner_model = wrapper_model.model  # Model wrapper 内部的 main_model
        else:
            inner_model = wrapper_model
        
        # 检查是否应该执行 push
        should_push = (
            hasattr(inner_model, 'push_prototypes') and
            (epoch + 1) >= push_start_epoch and
            (epoch + 1) <= push_end_epoch and
            (epoch + 1) % push_interval == 0 and
            not is_cooldown  # 关键: 避免在 cooldown 阶段 push
        )
        
        if should_push:
            if rank == 0:
                logger.info("---> Pushing prototypes at epoch {} (LR={:.6f}, momentum={})".format(
                    epoch + 1, current_lr, push_momentum))
            try:
                inner_model.push_prototypes(train_loader, gpu, momentum=push_momentum)
                if rank == 0:
                    logger.info("---> Prototype push completed (gradual update with momentum={})".format(push_momentum))
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Prototype push failed: {e}")
        elif hasattr(inner_model, 'push_prototypes') and (epoch + 1) % push_interval == 0:
            # 记录为什么跳过 push
            if rank == 0:
                skip_reason = []
                if (epoch + 1) < push_start_epoch:
                    skip_reason.append(f"epoch < push_start({push_start_epoch})")
                if (epoch + 1) > push_end_epoch:
                    skip_reason.append(f"epoch > push_end({push_end_epoch})")
                if is_cooldown:
                    skip_reason.append(f"cooldown phase (LR={current_lr:.6f})")
                if skip_reason:
                    logger.info("---> Skipping prototype push: {}".format(", ".join(skip_reason)))

        # ----------------------------------------- evaluate --------------------------------------- #
        is_best = False
        if active_eval:
            model.eval()
            eval_top1, eval_top3, precision, recall, f1, specificity, loss = run_eval(model, eval_loader)
            if rank == 0:
                logger.info(
                    "\tEval-Epoch: [{}/{}], Top1:{:.3f}, Top3:{:.3f}, Test_precision:{:.4f},Test_recall:{:.4f},Test_f1:{:.4f},,Test_specificity:{:.4f} Test_loss:{:.6f}".format(
                        epoch + 1, exp.max_epoch, eval_top1, eval_top3,precision,
                        recall,f1,specificity, loss
                    )
                )
                if eval_top1 > BEST_TOP1_ACC:
                    BEST_TOP1_ACC = eval_top1
                    BEST_TOP3_ACC = eval_top3
                    BEST_TOP1_EPOCH = epoch + 1
                    is_best = True
                logger.info(
                    "\tBest Top1 at epoch [{}/{}], Top1:{:.3f}, Top-3:{:.3f}".format(
                        BEST_TOP1_EPOCH, exp.max_epoch, BEST_TOP1_ACC, BEST_TOP3_ACC
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar("Val/Top1", eval_top1, (epoch + 1) * ITERS_PER_EPOCH)
                    tb_writer.add_scalar("Val/Top3", eval_top3, (epoch + 1) * ITERS_PER_EPOCH)

        # ----------------------------------------- dump weights ----------------------------------- #
        if rank == 0 and (is_best or (epoch + 1) % exp.dump_interval == 0 or (epoch + 1) == exp.max_epoch):
            exp.before_save_checkpoint()
            ckpt = {
                "start_epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_top1": BEST_TOP1_ACC,
                "best_top3": BEST_TOP3_ACC,
                "best_top1_epoch": BEST_TOP1_EPOCH,
                "scaler": scaler.state_dict()
                # 'temp': json.dumps(temp.cpu().detach().numpy()[0].tolist()),
                # 'decay_value': decay_value
            }
            save_checkpoint(ckpt, is_best, file_name, "last_epoch") \
                # # 每个 epoch 结束时打印准确度
        # if active_eval:
        #     model.eval()
        #     eval_top1, eval_top5 = run_eval(model, eval_loader)
        #     if rank == 0:
        #         logger.info(
        #             "Epoch: [{}/{}], Training Accuracy - Top1: {:.3f}, Top5: {:.3f}".format(
        #                 epoch + 1, exp.max_epoch, eval_top1, eval_top5
        #             )
        #         )
        #         if tb_writer:
        #             tb_writer.add_scalar("Accuracy/Top1", eval_top1, (epoch + 1) * ITERS_PER_EPOCH)
        #             tb_writer.add_scalar("Accuracy/Top5", eval_top5, (epoch + 1) * ITERS_PER_EPOCH)
    # ---------------------------------------- end of the training -------------------------------- #
    if rank == 0:
        logger.info("Training of experiment: {} is done.".format(exp.exp_name))
        if active_eval:
            logger.info(
                "\tBest Top1 at epoch [{}/{}], Top1:{:.3f}, Top3:{:.3f}".format(
                    BEST_TOP1_EPOCH, exp.max_epoch, BEST_TOP1_ACC, BEST_TOP3_ACC
                )
            )
        logger.stop()


if __name__ == "__main__":
    main()
