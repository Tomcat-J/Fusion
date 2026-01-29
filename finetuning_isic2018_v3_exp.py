# --------------------------------------------------------
# ISIC2018 数据集训练配置 V3 - 从已有 ISIC2018 权重继续
# 
# 【背景】
# - 你有一个在 ISIC2018 上训练好的权重（原始单原型头，85.52%）
# - 现在想用 HeadOptimal（多原型头）+ EnhancementModule 超过 85.52%
# 
# 【问题分析】
# - 加载权重时，backbone 成功加载，但 HeadOptimal 是随机初始化的
# - Epoch 1 = 82.54%（只靠 backbone 特征）
# - 训练后下降到 79-80%（学习率太高，破坏了 backbone）
# 
# 【V3 策略】
# - 长时间冻结 backbone（保护 ISIC2018 特征）
# - 只训练 HeadOptimal 和 Enhancement
# - 后期再用很低的学习率微调 backbone
# --------------------------------------------------------
import os
import torch
from loguru import logger
from projects.eval_tools.finetuning_stage2_exp import Exp as BaseExp


class Exp(BaseExp):
    """
    ISIC2018 训练配置 V3 - 从已有权重继续
    
    关键策略：
    1. 长时间冻结 backbone（20 epochs）
    2. 先让 HeadOptimal 学好
    3. 解冻后用极低学习率微调 backbone
    """
    def __init__(self, batch_size, max_epoch=100):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 数据集配置 =====
        self.dataset = "CommonDataSet"
        self.num_classes = 7
        
        # ===== 使用增强版模型 =====
        self.use_enhanced_model = True
        self.enhancement_reduction = 0.0625
        
        # ===== 优化器设置 =====
        self.opt = "adamw"
        self.weight_decay = 0.03
        self.clip_grad = 1.0
        self.clip_mode = "norm"

        # ===== 数据增强 - 中等 =====
        self.color_jitter = 0.3
        self.aa = "rand-m7-mstd0.5-inc1"
        self.reprob = 0.2
        self.remode = "pixel"
        self.recount = 1
        self.mixup = 0.0
        self.cutmix = 0.0
        self.smoothing = 0.1
        self.drop_path = 0.1
        
        # ===== 学习率设置 =====
        # 关键：适中的基础学习率
        self.basic_lr_per_img = 8e-5 / 32
        
        # ===== 分层学习率 - 保护 backbone =====
        self.backbone_lr_scale = 0.01     # 极低！保护 ISIC2018 特征
        self.mhf_lr_scale = 0.05          # MHF 也要保护
        self.enhancement_lr_scale = 2.0   # Enhancement 可以积极学习
        self.head_lr_scale = 1.5          # HeadOptimal 需要学习
        
        # ===== 学习率调度 =====
        self.sched = "warmcos"
        self.warmup_epochs = 10
        self.warmup_lr = 1e-6
        self.min_lr = 1e-7
        self.cooldown_epochs = 15

        # ===== 冻结策略 - 关键！=====
        # 长时间冻结 backbone，让 HeadOptimal 先学好
        self.freeze_backbone_epochs = 20

        # ===== 禁用 Prototype Push =====
        self.push_start_epoch = 999
        self.push_interval = 999
        self.push_end_epoch = 0
        self.push_momentum = 0.0

        # ===== EMA =====
        self.model_ema = True
        self.model_ema_decay = 0.9998

        # ===== 路径设置 =====
        self.pretrain_exp_name = "HiFuse_Small_ISIC2018_v3_1"
        self.save_folder_prefix = "ft_isic2018_v3_1"
        
        # ===== 损失函数权重 - 温和 =====
        self.loss_weights = {
            'clst_scale': 0.5,
            'sep_scale': 0.15,
            'orth_scale': 0.01,
            'local_consistency_scale': 0.05,
            'margin': 0.3,
        }

    def get_model(self):
        """重写 get_model"""
        model = super().get_model()
        
        if hasattr(model.model, 'head') and hasattr(model.model.head, 'proto_loss_fn'):
            loss_fn = model.model.head.proto_loss_fn
            loss_fn.clst_scale = self.loss_weights['clst_scale']
            loss_fn.sep_scale = self.loss_weights['sep_scale']
            loss_fn.orth_scale = self.loss_weights['orth_scale']
            loss_fn.local_consistency_scale = self.loss_weights['local_consistency_scale']
            loss_fn.margin = self.loss_weights['margin']
            
            logger.info(f"[ISIC2018-V3] Backbone frozen for {self.freeze_backbone_epochs} epochs")
            logger.info(f"[ISIC2018-V3] Backbone LR scale: {self.backbone_lr_scale}")
        
        return model


class ExpHeadOnly(Exp):
    """
    只训练 Head 的配置（完全冻结 backbone）
    
    使用场景：
    - 如果 V3 效果不好，可以先只训练 HeadOptimal
    - 然后用训练好的 HeadOptimal 权重再微调全模型
    """
    def __init__(self, batch_size, max_epoch=50):
        super(ExpHeadOnly, self).__init__(batch_size, max_epoch)
        
        # 完全冻结 backbone
        self.freeze_backbone_epochs = 999
        
        # 更高的 head 学习率
        self.head_lr_scale = 2.0
        self.enhancement_lr_scale = 3.0
        
        self.save_folder_prefix = "ft_isic2018_headonly_"


class ExpFineTune(Exp):
    """
    微调配置（从 HeadOnly 权重继续）
    
    使用方法：
    1. 先用 ExpHeadOnly 训练 50 epochs
    2. 用最佳权重继续，使用这个配置微调全模型
    """
    def __init__(self, batch_size, max_epoch=50):
        super(ExpFineTune, self).__init__(batch_size, max_epoch)
        
        # 短冻结期
        self.freeze_backbone_epochs = 5
        
        # 更低的学习率
        self.basic_lr_per_img = 3e-5 / 32
        self.backbone_lr_scale = 0.005  # 极低
        
        self.save_folder_prefix = "ft_isic2018_finetune_"


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 70)
    print("ISIC2018 训练配置 V3 - 从已有权重继续")
    print("=" * 70)
    print(f"类别数: {exp.num_classes}")
    print(f"基线准确率: 85.52% (原始单原型头)")
    print(f"目标: 使用 HeadOptimal 超过 85.52%")
    print("=" * 70)
    print("\n关键设置:")
    print(f"  - Backbone 冻结: {exp.freeze_backbone_epochs} epochs")
    print(f"  - Backbone LR scale: {exp.backbone_lr_scale}")
    print(f"  - Head LR scale: {exp.head_lr_scale}")
    print("=" * 70)
    print("\n使用方法:")
    print("python train.py --exp_file finetuning_isic2018_v3_exp.py \\")
    print("    --ckpt <你的ISIC2018权重>")
    print("=" * 70)
