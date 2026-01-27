# --------------------------------------------------------
# ISIC2018 数据集训练配置
# 
# 【数据集特点】
# - 7 类皮肤病变分类
# - 基线准确率 85.52%（有提升空间）
# - 类别不平衡问题
# 
# 【策略】
# - 积极策略，充分利用增强模块
# - 较高学习率，快速收敛
# - 增强类间分离
# --------------------------------------------------------
import os
import torch
from loguru import logger
from projects.eval_tools.finetuning_stage2_exp import Exp as BaseExp


class Exp(BaseExp):
    """
    ISIC2018 训练配置 - 积极策略
    """
    def __init__(self, batch_size, max_epoch=100):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 数据集配置 =====
        self.dataset = "CommonDataSet"
        self.num_classes = 7  # ISIC2018 有 7 个类别
        
        # ===== 使用增强版模型 =====
        self.use_enhanced_model = True
        self.enhancement_reduction = 0.0625
        
        # ===== 优化器设置 =====
        self.opt = "adamw"
        self.weight_decay = 0.05
        self.clip_grad = 1.0
        self.clip_mode = "norm"

        # ===== 数据增强 - 较强（提升泛化） =====
        self.color_jitter = 0.4
        self.aa = "rand-m9-mstd0.5-inc1"
        self.reprob = 0.25
        self.remode = "pixel"
        self.recount = 1
        self.mixup = 0.0              # 禁用 mixup（与多原型头不兼容）
        self.cutmix = 0.0
        self.smoothing = 0.1
        self.drop_path = 0.15         # 略高
        
        # ===== 学习率设置 - 较高（有提升空间） =====
        self.basic_lr_per_img = 1e-4 / 32
        
        # ===== 分层学习率 =====
        self.backbone_lr_scale = 0.05     # 释放 backbone
        self.mhf_lr_scale = 0.2           # MHF 积极
        self.enhancement_lr_scale = 3.0   # 增强模块高学习率
        self.head_lr_scale = 2.0          # head 积极
        
        # ===== 学习率调度 =====
        self.sched = "warmcos"
        self.warmup_epochs = 5
        self.warmup_lr = 1e-6
        self.min_lr = 5e-7
        self.cooldown_epochs = 10

        # ===== 冻结策略 =====
        self.freeze_backbone_epochs = 3   # 短冻结期

        # ===== 禁用 Prototype Push =====
        self.push_start_epoch = 999
        self.push_interval = 999
        self.push_end_epoch = 0
        self.push_momentum = 0.0

        # ===== EMA =====
        self.model_ema = True
        self.model_ema_decay = 0.9998

        # ===== 路径设置 =====
        self.pretrain_exp_name = "HiFuse_Small_ISIC2018"
        self.save_folder_prefix = "ft_isic2018_"
        
        # ===== 损失函数权重 - 增强类间分离 =====
        self.loss_weights = {
            'clst_scale': 0.8,
            'sep_scale': 0.25,         # 增强类间分离
            'orth_scale': 0.02,
            'local_consistency_scale': 0.1,
            'margin': 0.4,             # 较大 margin
        }

    def get_model(self):
        """重写 get_model，动态修改损失权重"""
        model = super().get_model()
        
        if hasattr(model.model, 'head') and hasattr(model.model.head, 'proto_loss_fn'):
            loss_fn = model.model.head.proto_loss_fn
            loss_fn.sep_scale = self.loss_weights['sep_scale']
            loss_fn.orth_scale = self.loss_weights['orth_scale']
            loss_fn.local_consistency_scale = self.loss_weights['local_consistency_scale']
            loss_fn.margin = self.loss_weights['margin']
            
            logger.info(f"[ISIC2018] Loss weights adjusted for 7-class classification")
        
        return model


class ExpAggressive(Exp):
    """
    更激进的配置（如果默认效果不够好）
    """
    def __init__(self, batch_size, max_epoch=100):
        super(ExpAggressive, self).__init__(batch_size, max_epoch)
        
        self.basic_lr_per_img = 1.5e-4 / 32
        self.backbone_lr_scale = 0.08
        self.mhf_lr_scale = 0.3
        self.head_lr_scale = 2.5
        
        self.loss_weights['sep_scale'] = 0.3
        self.loss_weights['margin'] = 0.5
        
        self.save_folder_prefix = "ft_isic2018_aggressive_"


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 70)
    print("ISIC2018 训练配置 - 积极策略")
    print("=" * 70)
    print(f"类别数: {exp.num_classes}")
    print(f"基线准确率: 85.52%")
    print(f"目标: 突破 88%+")
    print("=" * 70)
