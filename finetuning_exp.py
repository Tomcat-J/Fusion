# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# 高精度训练配置 - 100 epochs 目标 90% 准确率
# 
# 【数据集特点】
# - 3 类肺结节分类
# - 1753 训练样本，468 验证样本
# - 小数据集，容易过拟合
# 
# 【策略 - 100 epochs 优化版】
# 1. 使用原始模型 (main_model + Head) 确保稳定性
# 2. 更高的学习率 + 更短的 warmup (快速收敛)
# 3. 适度的数据增强 + 正则化
# 4. 更早、更频繁的 prototype push
# 5. 使用 EMA 平滑模型参数
# --------------------------------------------------------
from projects.eval_tools.finetuning_mae_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=100):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 【关键】使用原始模型确保稳定性 =====
        self.use_enhanced_model = False
        
        # ===== 优化器设置 =====
        self.opt = "adamw"
        self.weight_decay = 0.05  # 稍强的权重衰减
        self.clip_grad = 1.0      # 梯度裁剪
        self.clip_mode = "norm"

        # ===== 数据增强 - 适度增强 =====
        self.color_jitter = 0.4      # 颜色抖动
        self.aa = "rand-m9-mstd0.5-inc1"  # AutoAugment
        self.reprob = 0.25           # Random Erasing
        self.remode = "pixel"
        self.recount = 1
        
        # 禁用 mixup/cutmix (与多原型头不兼容)
        self.mixup = 0.0
        self.cutmix = 0.0
        
        # 正则化
        self.smoothing = 0.1         # Label Smoothing
        self.drop_path = 0.1         # DropPath
        
        # ===== 学习率设置 - 100 epochs 优化 =====
        # 使用稍高的学习率，加速收敛
        self.basic_lr_per_img = 2e-4 / 32  # base LR = 2e-4 (比之前高)
        
        # ===== 学习率调度 - 100 epochs 优化 =====
        self.sched = "warmcos_scale"
        self.warmup_epochs = 5       # 短 warmup (5% of total)
        self.warmup_lr = 1e-6        # warmup 起始 LR
        self.min_lr = 1e-6           # 最小 LR
        self.cooldown_epochs = 10    # 短 cooldown (10% of total)

        # ===== 差异化学习率 (原始模型不使用，但保留配置) =====
        self.backbone_lr_scale = 0.1
        self.mhf_lr_scale = 0.5
        self.enhancement_lr_scale = 1.0
        
        # ===== Prototype Push 策略 - 100 epochs 优化 =====
        # 更早开始、更频繁的 push
        self.push_start_epoch = 15   # 更早开始 push
        self.push_interval = 10      # 每 10 epochs push 一次
        self.push_end_epoch = 90     # 最后 10 epochs 不 push
        self.push_momentum = 0.9     # push 动量
        
        # ===== 不冻结 backbone =====
        self.freeze_backbone_epochs = 0

        # ===== EMA =====
        self.model_ema = True        # 启用 EMA
        self.model_ema_decay = 0.9998  # 稍低的 decay (更快适应)
        self.model_ema_force_cpu = False

        # ===== 其他设置 =====
        self.weights_prefix = ""
        self.pretrain_exp_name = "HiFuse_Small_1e-5-0.05"
        self.save_folder_prefix = "ft_100e_"
        self.print_interval = 10
        self.dump_interval = 10
        self.eval_interval = 5  # 每 5 epochs 评估一次


# ===== 备选配置：更激进的策略 =====
class ExpAggressive(BaseExp):
    """
    更激进的 100 epochs 配置
    适合在基础配置达到 85%+ 后尝试
    """
    def __init__(self, batch_size, max_epoch=100):
        super(ExpAggressive, self).__init__(batch_size, max_epoch)

        self.use_enhanced_model = False
        
        # 更强的优化器设置
        self.opt = "adamw"
        self.weight_decay = 0.08  # 更强的权重衰减
        self.clip_grad = 0.5     # 更严格的梯度裁剪
        self.clip_mode = "norm"

        # 更强的数据增强
        self.color_jitter = 0.5
        self.aa = "rand-m10-mstd0.5-inc1"
        self.reprob = 0.3
        self.mixup = 0.0
        self.cutmix = 0.0
        self.smoothing = 0.15
        self.drop_path = 0.15
        
        # 学习率
        self.basic_lr_per_img = 1.5e-4 / 32
        
        # 调度
        self.warmup_epochs = 5
        self.warmup_lr = 1e-7
        self.min_lr = 1e-7
        self.cooldown_epochs = 15

        # 更频繁的 push
        self.push_start_epoch = 10
        self.push_interval = 8
        self.push_end_epoch = 90
        self.push_momentum = 0.92
        
        self.freeze_backbone_epochs = 0
        self.model_ema = True
        self.model_ema_decay = 0.9995

        self.pretrain_exp_name = "HiFuse_Small_1e-5-0.05"
        self.save_folder_prefix = "ft_100e_aggressive_"
        self.eval_interval = 5


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 60)
    print("高精度训练配置 - 100 epochs 目标 90% 准确率")
    print("=" * 60)
    
    # 验证学习率计算
    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"\n【学习率配置】")
    print(f"基础学习率: {exp.lr:.6f}")
    print(f"warmup_lr: {exp.warmup_lr}")
    print(f"min_lr: {exp.min_lr}")
    
    print(f"\n【训练周期】")
    print(f"总 epochs: {exp.max_epoch}")
    print(f"warmup: {exp.warmup_epochs} epochs")
    print(f"cooldown: {exp.cooldown_epochs} epochs")
    
    print(f"\n【数据增强】")
    print(f"color_jitter: {exp.color_jitter}")
    print(f"reprob: {exp.reprob}")
    print(f"smoothing: {exp.smoothing}")
    print(f"drop_path: {exp.drop_path}")
    
    print(f"\n【正则化】")
    print(f"weight_decay: {exp.weight_decay}")
    print(f"model_ema: {exp.model_ema}")
    
    print(f"\n【Prototype Push】")
    print(f"开始: epoch {exp.push_start_epoch}")
    print(f"结束: epoch {exp.push_end_epoch}")
    print(f"间隔: {exp.push_interval} epochs")
    print(f"动量: {exp.push_momentum}")
    
    print("\n" + "=" * 60)
    print("100 epochs 优化策略:")
    print("1. 更高的学习率 (2e-4) 加速收敛")
    print("2. 短 warmup (5 epochs) 快速进入训练")
    print("3. 更早的 Prototype Push (epoch 15 开始)")
    print("4. 更频繁的 Push (每 10 epochs)")
    print("5. EMA 平滑 (decay=0.9998)")
    print("=" * 60)
