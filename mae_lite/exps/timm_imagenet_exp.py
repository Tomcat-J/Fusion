# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
Support timm training (timm==0.4.12)
Differences:
* lr, warmup_lr, min_lr are replaced by basic_lr_per_img, warmup_lr_per_img, min_lr_per_img
* Some features are not supported now, eg., aug_repeats, jsd_loss, bce_loss, split_bn
"""
import os
import torch
from torch import nn

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm import create_model
from timm.data import create_loader, resolve_data_config, Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler as create_scheduler_timm
from timm.optim import create_optimizer
from timm.utils import ModelEmaV2

from mae_lite.data.datasets.commondata import read_train_data, read_val_data, CommonDataSet
from mae_lite.data.datasets.lung import read_split_data, transform
from mae_lite.exps.base_exp import BaseExp
from mae_lite.data import build_dataset
from mae_lite.data.transforms import transforms, ToRGB
from mae_lite.layers.lr_scheduler import LRScheduler, _Scheduler
import mae_lite.utils.torch_dist as dist
import numpy as np
import torch.nn.functional as F


# def focal_loss(labels, logits, alpha, gamma):
#     """Compute the focal loss between `logits` and the ground truth `labels`.

#     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

#     Args:
#       labels: A float tensor of size [batch, num_classes].
#       logits: A float tensor of size [batch, num_classes].
#       alpha: A float tensor of size [batch_size]
#         specifying per-example weight for balanced cross entropy.
#       gamma: A float scalar modulating loss from hard and easy examples.

#     Returns:
#       focal_loss: A float32 scalar representing normalized total loss.
#     """
#     BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

#     if gamma == 0.0:
#         modulator = 0.5
#     else:
#         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +torch.exp(-1.0 * logits)))

#     loss = modulator * BCLoss

#     weighted_loss = alpha * loss
#     focal_loss = torch.sum(weighted_loss)

#     focal_loss /= torch.sum(labels)
#     return focal_loss



# def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
#     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

#     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
#     where Loss is one of the standard losses used for Neural Networks.

#     Args:
#       labels: A int tensor of size [batch].
#       logits: A float tensor of size [batch, no_of_classes].
#       samples_per_cls: A python list of size [no_of_classes].
#       no_of_classes: total number of classes. int
#       loss_type: string. One of "sigmoid", "focal", "softmax".
#       beta: float. Hyperparameter for Class balanced loss.
#       gamma: float. Hyperparameter for Focal loss.

#     Returns:
#       cb_loss: A float tensor representing class balanced loss
#     """
#     effective_num = 1.0 - np.power(beta, samples_per_cls)
#     weights = (1.0 - beta) / np.array(effective_num)
#     weights = weights / np.sum(weights) * no_of_classes
#     entroy_weights = weights
#     entroy_weights = torch.tensor(entroy_weights).float()
#     entroy_weights = entroy_weights.cuda(non_blocking=True)
#     labels_one_hot = F.one_hot(labels, no_of_classes).float()

#     weights = torch.tensor(weights).float()
#     weights = weights.unsqueeze(0)
#     weights = weights.cuda(non_blocking=True)
#     weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
#     weights = weights.sum(1)
#     weights = weights.unsqueeze(1)
#     weights = weights.repeat(1,no_of_classes)

#     if loss_type == "focal":
#         cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
#     elif loss_type == "sigmoid":
#         cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
#     elif loss_type == "softmax":
#         pred = logits.softmax(dim = 1)
#         cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
#     elif loss_type == "entroy_cross":
#         cb_loss = F.cross_entropy(logits, labels, weight=entroy_weights)
#     return cb_loss


class Model(nn.Module):
    def __init__(self, args, model):
        super(Model, self).__init__()
        self.model = model
        # mixup
        mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes,
            )
        else:
            mixup_fn = None
        self.mixup_fn = mixup_fn

        # criterion
        if mixup_active:
            train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        self.train_loss_fn = train_loss_fn
        # ema
        if args.model_ema:
            self.ema_model = ModelEmaV2(
                self.model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else None
            )
            for p in self.ema_model.parameters():
                p.requires_grad = False
        else:
            self.ema_model = None
        
        # 检测模型类型: 多原型头 vs 标准线性头
        self._use_multi_prototype_head = self._detect_multi_prototype_head()
    
    def _detect_multi_prototype_head(self) -> bool:
        """检测内部模型是否使用多原型分类头"""
        if hasattr(self.model, 'head'):
            head = self.model.head
            # 检查是否有 proto_manager 属性
            if hasattr(head, 'proto_manager'):
                return True
        return False

    # def forward(self, x, target=None,epoch = None, update_param=False):
    #     if self.training:
    #         # target_ce = target
    #         if self.mixup_fn is not None:
    #             x, target = self.mixup_fn(x, target)
    #         logits,head_loss = self.model(x)
    #         loss = self.train_loss_fn(logits, target) + head_loss
    #         # loss = self.compute_loss(logits,epoch,target_ce,target) + head_loss
    #         if self.ema_model is not None:
    #             self.ema_model.update(self.model)
    #
    #         # TODO: accuracy monitor
    #         # top1, top5 = accuracy(logits, target, (1, 5))
    #         return loss, None
    #     else:
    #         logits = self.model(x)
    #         return logits


    def forward(self, x, target=None, epoch=None, update_param=False):
        """
        前向传播 - 同时支持多原型分类头和标准线性头
        
        多原型头模型 (main_model 等):
            - 训练时: 内部模型返回 (loss, extra_dict)，loss 已包含 CE + proto_loss
            - 评估时: 内部模型返回 (logits, head_loss)
        
        标准线性头模型 (CNN, ViT, ConvNeXt baseline 等):
            - 训练时: 内部模型返回 logits，使用 train_loss_fn 计算损失
            - 评估时: 内部模型返回 logits
        
        【修复】Mixup 标签逻辑:
            - 多原型头: 使用原始硬标签 (target_orig)，因为原型损失需要硬标签
            - 标准线性头: 使用 mixup 后的软标签 (target)
        """
        if self.training:
            target_orig = target  # 保存原始硬标签
            
            if self.mixup_fn is not None:
                x, target = self.mixup_fn(x, target)  # target 变成软标签
            
            if self._use_multi_prototype_head:
                # 多原型头模型: 
                # 【重要】多原型头需要硬标签来计算原型损失 (cluster, separation 等)
                # 但 Mixup 后的图像确实是混合的，这里有两种策略:
                # 策略1: 禁用 Mixup (推荐，因为原型学习与 Mixup 理论上不兼容)
                # 策略2: 使用原始标签，接受理论上的不一致
                # 
                # 当前实现: 如果启用了 Mixup，打印警告并使用原始标签
                if self.mixup_fn is not None:
                    import warnings
                    warnings.warn(
                        "Mixup is enabled with multi-prototype head. "
                        "This may cause training instability. "
                        "Consider disabling Mixup (mixup=0, cutmix=0) for prototype-based models.",
                        UserWarning
                    )
                loss, extra_dict = self.model(x, target=target_orig)
            else:
                # 标准线性头模型: forward 返回 logits
                output = self.model(x)
                
                # 处理不同的返回格式
                if isinstance(output, tuple):
                    logits = output[0]
                    head_loss = output[1] if len(output) > 1 else torch.tensor(0.0, device=x.device)
                else:
                    logits = output
                    head_loss = torch.tensor(0.0, device=x.device)
                
                # 使用 train_loss_fn 计算分类损失 (使用 mixup 后的软标签)
                loss = self.train_loss_fn(logits, target) + head_loss
                extra_dict = {'ce_loss': loss.detach(), 'head_loss': head_loss.detach() if isinstance(head_loss, torch.Tensor) else head_loss}
            
            if self.ema_model is not None:
                self.ema_model.update(self.model)
            
            return loss, extra_dict if extra_dict else {}
        else:
            # 评估模式
            output = self.model(x, target=target) if self._use_multi_prototype_head else self.model(x)
            
            if isinstance(output, tuple):
                logits = output[0]
                head_loss = output[1] if len(output) > 1 else torch.tensor(0.0, device=x.device)
            else:
                logits = output
                head_loss = torch.tensor(0.0, device=x.device)
            
            return logits, head_loss


    def compute_loss(self,logits, epoch ,targets_ce,targets):

        beta = 0.9999
        gamma_ce = 0.5
        loss_type = "focal"
        samples_per_cls = [1543, 196, 781]
        no_of_classes = 3
        loss_ce = (0.5 + epoch / 200) * CB_loss(targets_ce,logits, samples_per_cls, no_of_classes, loss_type,
                                                beta,
                                                gamma_ce) + (0.5 - epoch / 200) * self.train_loss_fn(logits,targets)

        return loss_ce


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)
        # dataset & model
        self.dataset = "MyDataSet"
        # self.dataset = "CIFAR100"
        self.encoder_arch = "HiFuse_Small"
        self.pretrained = False
        self.num_classes = 3
        self.global_pool = None
        self.img_size = None
        self.input_size = None
        self.crop_pct = None
        self.mean = None
        self.std = None
        self.interpolation = ""
        self.validation_batch_size = None
        self.validation_dataset = "MyDataSet"
        self.test_dataset = "MyDataSet"
        self.test_batch_size = None

        # optimizer
        self.opt = "sgd"
        self.opt_eps = None
        self.opt_betas = None
        self.momentum = 0.9
        self.weight_decay = 2e-5
        self.clip_grad = None
        self.clip_mode = "norm"

        # schedule
        self.sched = "cosine"
        # self.lr = 0.05
        self.basic_lr_per_img = 0.01 / batch_size
        self.lr_noise = None
        self.lr_noise_pct = 0.67
        self.lr_noise_std = 1.0
        self.lr_cycle_mul = 1.0
        self.lr_cycle_decay = 0.5
        self.lr_cycle_limit = 1
        self.lr_k_decay = 1.0
        # self.warmup_lr = 0.0001
        self.warmup_lr_per_img = 0.0001 / batch_size
        # self.min_lr = 1e-6
        self.min_lr_per_img = 1e-5 / batch_size
        self.epochs = max_epoch
        self.epoch_repeats = 0
        self.start_epoch = None  #
        self.decay_epochs = None
        self.warmup_epochs = 3
        self.cooldown_epochs = 10
        self.patience_epochs = 10
        self.decay_rate = 0.1

        # augmentation & regularization
        self.no_aug = False
        self.scale = (0.08, 1.0)
        self.ratio = (3.0 / 4, 4.0 / 3.0)
        self.hflip = 0.5
        self.vflip = 0.0
        self.color_jitter = 0.4
        self.aa = None
        # self.aug_repeats = 0  # not support
        # self.aug_splits = 0  # not support
        # self.jsd_loss = False  # not support
        # self.bce_loss = False  # not support
        self.reprob = 0.0
        self.remode = "pixel"
        self.recount = 1
        self.resplit = False
        self.mixup = 0.0
        self.cutmix = 0.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = "batch"
        self.mixup_off_epoch = 0
        self.smoothing = 0.0
        self.train_interpolation = "bicubic"
        self.drop = 0.0
        self.drop_connect = None
        self.drop_path = None
        self.drop_block = None

        # batch norm
        self.bn_tf = False
        self.bn_momentum = None
        self.bn_eps = None
        self.sync_bn = False
        self.dist_bn = "reduce"
        # self.split_bn = False  # not support

        # EMA
        self.model_ema = False
        self.model_ema_force_cpu = False
        self.model_ema_decay = 0.9998

        self.seed = 0
        self.num_workers = 10
        self.weights_prefix = ""
        self.print_interval = 10
        self.enable_tensorboard = True
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]

    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(
                self.encoder_arch,
                pretrained=self.pretrained,
                num_classes=self.num_classes,
                drop_rate=self.drop,
                drop_connect_rate=self.drop_connect,
                drop_path_rate=self.drop_path,
                drop_block_rate=self.drop_block,
                global_pool=self.global_pool,
                bn_tf=self.bn_tf,
                bn_momentum=self.bn_momentum,
                bn_eps=self.bn_eps,
            )
            if self.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = Model(self, model)
        return self.model

    def get_data_loader(self):
        if "data_loader" not in self.__dict__:
            img_size = 224
            data_transform = {
                "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5234546, 0.3153837, 0.29106084], [0.20845978, 0.18016016, 0.16627173])]),
                "val": transforms.Compose([transforms.Resize(img_size),
                                           transforms.CenterCrop(img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5234546, 0.3153837, 0.29106084], [0.20845978, 0.18016016, 0.16627173])])}
            # train_images_path, train_images_label = read_train_data("/home/backup/lh/KD/data/ISIC2018_Task3/train")
            # val_images_path, val_images_label = read_val_data("/home/backup/lh/KD/data/ISIC2018_Task3/test")
            # dataset_train = build_dataset(self.dataset,images_path=train_images_path,
            #                   images_class=train_images_label,
            #                   transform=data_transform["train"])
            # dataset_eval = build_dataset(self.validation_dataset if self.validation_dataset else self.dataset, images_path=val_images_path,
            #                 images_class=val_images_label,
            #                 transform=data_transform["val"])
            train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
                "/home/backup/lh/lung/data/kd3/test", "/home/backup/lh/lung/data/super_no_aug",
                "/home/backup/lh/lung/data/kd3/val")
            dataset_train = build_dataset(self.dataset,images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
            dataset_eval = build_dataset(self.validation_dataset if self.validation_dataset else self.dataset, images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
            # dataset_test = build_dataset(self.test_dataset if self.test_dataset else self.dataset, images_path=test_images_path,
            #                 images_class=test_images_label,
            #                 transform=transform("test"))
            # dataset_train = build_dataset(self.dataset, True)
            # dataset_eval = build_dataset(self.validation_dataset if self.validation_dataset else self.dataset, False)
            batch_size_per_gpu = self.batch_size // dist.get_world_size()
            data_config = resolve_data_config(vars(self), model=self.get_model().model, verbose=dist.is_main_process())
            loader_train = create_loader(
                dataset_train,
                input_size=data_config["input_size"],
                batch_size=batch_size_per_gpu,
                is_training=True,
                use_prefetcher=False,
                no_aug=self.no_aug,
                re_prob=self.reprob,
                re_mode=self.remode,
                re_count=self.recount,
                re_split=self.resplit,
                scale=self.scale,
                ratio=self.ratio,
                hflip=self.hflip,
                vflip=self.vflip,
                color_jitter=self.color_jitter,
                auto_augment=self.aa,
                interpolation=self.train_interpolation,
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=self.num_workers,
                distributed=dist.is_distributed(),
                pin_memory=False,
            )
            validation_batch_size_per_gpu = (self.validation_batch_size or self.batch_size) // dist.get_world_size()
            loader_eval = create_loader(
                dataset_eval,
                input_size=data_config["input_size"],
                batch_size=validation_batch_size_per_gpu,
                is_training=False,
                use_prefetcher=False,
                interpolation=data_config["interpolation"],
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=self.num_workers,
                distributed=dist.is_distributed(),
                crop_pct=data_config["crop_pct"],
                pin_memory=False,
            )
            # test_batch_size_per_gpu = (self.test_batch_size or self.batch_size) // dist.get_world_size()
            # loader_test = create_loader(
            #     dataset_eval,
            #     input_size=data_config["input_size"],
            #     batch_size=test_batch_size_per_gpu,
            #     is_training=False,
            #     use_prefetcher=False,
            #     interpolation=data_config["interpolation"],
            #     mean=data_config["mean"],
            #     std=data_config["std"],
            #     num_workers=self.num_workers,
            #     distributed=dist.is_distributed(),
            #     crop_pct=data_config["crop_pct"],
            #     pin_memory=False,
            # )
            loader_train.dataset.transform = transforms.Compose([ToRGB(), loader_train.dataset.transform])
            loader_eval.dataset.transform = transforms.Compose([ToRGB(), loader_eval.dataset.transform])
            # loader_test.dataset.transform = transforms.Compose([ToRGB(), loader_test.dataset.transform])
            self.data_loader = {"train": loader_train, "eval": loader_eval}
        return self.data_loader

    def get_optimizer(self):
        if "optimizer" not in self.__dict__:
            if "lr" not in self.__dict__:
                self.lr = self.basic_lr_per_img * self.batch_size
            self.optimizer = create_optimizer(self, self.get_model())
        return self.optimizer

    def get_lr_scheduler(self):
        if "lr" not in self.__dict__:
            self.lr = self.basic_lr_per_img * self.batch_size
        if "warmup_lr" not in self.__dict__:
            self.warmup_lr = self.warmup_lr_per_img * self.batch_size
        if "min_lr" not in self.__dict__:
            self.min_lr = self.min_lr_per_img * self.batch_size
        if "epochs" not in self.__dict__:
            self.epochs = self.max_epoch
        optimizer = self.get_optimizer()
        iters_per_epoch = len(self.get_data_loader()["train"])
        scheduler = TimmLRScheduler(self, optimizer, interval=iters_per_epoch)
        return scheduler


class TimmLRScheduler(LRScheduler):
    def __init__(self, args, optimizer, interval=1):
        self.scheduler_timm, _ = create_scheduler_timm(args, optimizer)
        super(TimmLRScheduler, self).__init__(optimizer, _Scheduler(0.0, 1), interval)

    def step(self, count):
        count, inner_count = divmod(count, self.interval)
        if inner_count == 0:
            self.scheduler_timm.step(count)

    def state_dict(self):
        return self.scheduler_timm.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler_timm.load_state_dict(state_dict)
