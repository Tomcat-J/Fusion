# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# 最优训练配置 - HiFuse + MHF Enhancement + HeadOptimal
# 
# 核心优化:
# 1. 学习率修正: basic_lr_per_img = 1e-4 / 32 (实际 LR = 1e-4)
# 2. 差异化学习率: backbone 0.1x, mhf 0.5x, enhancement 1.0x, head 2.0x
# 3. Prototype Push 优化: 避免 cooldown 阶段 push，使用渐进式动量更新
# 4. Head 预热: 前 5 epoch 冻结 backbone
# 5. 禁用 Mixup/CutMix: 与多原型头不兼容
# --------------------------------------------------------
from projects.eval_tools.finetuning_mae_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 优化器设置 =====
        self.clip_grad = 1.0  # 启用梯度裁剪，防止梯度爆炸
        self.clip_mode = "norm"

        # ===== 数据增强 & 正则化 =====
        # 【重要】禁用 Mixup/CutMix，因为与多原型头不兼容
        self.color_jitter = 0.3
        self.aa = "rand-m9-mstd0.5-inc1"  # 适度的 AutoAugment
        self.reprob = 0.1  # 轻微的 Random Erasing
        self.mixup = 0.0   # 【禁用】与多原型头不兼容
        self.cutmix = 0.0  # 【禁用】与多原型头不兼容
        self.smoothing = 0.1  # 标签平滑，帮助训练稳定
        self.drop_path = 0.1  # 轻微的 DropPath
        
        # ===== MHF Enhancement 设置 =====
        self.use_enhanced_model = True  # 启用 MHF 增强模块
        self.enhancement_reduction = 0.0625  # 增强模块通道缩减比例 (1/16)
        
        # ===== 差异化学习率设置 =====
        # 针对 ImageNet → 医学图像 迁移学习场景优化
        self.backbone_lr_scale = 0.1   # backbone: 低学习率，保护预训练特征
        self.mhf_lr_scale = 0.5        # MHF 原始参数: 中等学习率
        self.enhancement_lr_scale = 1.0  # 增强模块: 标准学习率 (从头训练)
        self.head_lr_scale = 2.0       # Head: 高学习率，加速收敛
        
        # ===== Head 预热设置 =====
        self.freeze_backbone_epochs = 5  # 前 5 epoch 冻结 backbone，只训练 Head
        
        # ===== Prototype Push 优化策略 =====
        # 核心改进: 避免在 cooldown 阶段 push，使用渐进式更新
        self.push_start_epoch = 15     # 延迟 push 开始 (等 Head 稳定)
        self.push_interval = 10        # push 间隔
        self.push_end_epoch = max_epoch - 20  # 在最后 20 epoch 前停止 push
        self.push_momentum = 0.9       # push 动量 (渐进式更新)
        self.push_lr_recovery_epochs = 3  # push 后学习率恢复 epoch 数
        self.push_lr_recovery_scale = 1.5  # push 后学习率恢复倍率
        
        # ===== 学习率调度优化 =====
        self.warmup_epochs = 10  # warmup epoch 数
        self.min_lr = 1e-5       # 最小学习率 (避免 cooldown 时 LR=0)
        self.cooldown_epochs = 10  # cooldown epoch 数

        # ===== 其他设置 =====
        self.weights_prefix = ""
        self.save_folder_prefix = "ft_optimal_"
        self.print_interval = 10
        self.dump_interval = 10
        self.eval_interval = 1  # 每个 epoch 都评估


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 60)
    print("最优训练配置验证")
    print("=" * 60)
    
    # 验证学习率计算
    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"基础学习率: {exp.lr:.6f}")
    print(f"  - backbone: {exp.lr * exp.backbone_lr_scale:.6f}")
    print(f"  - mhf: {exp.lr * exp.mhf_lr_scale:.6f}")
    print(f"  - enhancement: {exp.lr * exp.enhancement_lr_scale:.6f}")
    print(f"  - head: {exp.lr * exp.head_lr_scale:.6f}")
    
    print(f"\nPrototype Push 策略:")
    print(f"  - 开始 epoch: {exp.push_start_epoch}")
    print(f"  - 结束 epoch: {exp.push_end_epoch}")
    print(f"  - 间隔: {exp.push_interval}")
    print(f"  - 动量: {exp.push_momentum}")
    
    print(f"\n数据增强:")
    print(f"  - Mixup: {exp.mixup} (禁用)")
    print(f"  - CutMix: {exp.cutmix} (禁用)")
    print(f"  - Label Smoothing: {exp.smoothing}")
    
    model = exp.get_model()
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
