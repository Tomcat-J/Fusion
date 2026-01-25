# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# 最优训练配置 V3 - HiFuse + MHF Enhancement + HeadOptimal
# 
# 【V3 关键修复】基于 log3.txt 分析:
# 问题: CE Loss 始终 ~1.098 (= log(3))，模型没有学到分类能力
# 原因: backbone LR = 1e-6 太低，特征无法适配医学图像
# 
# 解决方案:
# 1. 提高 backbone_lr_scale 到 0.1 (原 0.01)
# 2. 提高 base LR 到 5e-4 (原 1e-4)
# 3. 缩短 warmup 到 5 epochs (原 20)
# 4. 更早开始 push (epoch 10)
# --------------------------------------------------------
from projects.eval_tools.finetuning_mae_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 优化器设置 =====
        self.clip_grad = 1.0  # 启用梯度裁剪，防止梯度爆炸
        self.clip_mode = "norm"

        # ===== 数据增强 & 正则化 =====
        self.color_jitter = 0.3
        self.aa = "rand-m9-mstd0.5-inc1"
        self.reprob = 0.1
        self.mixup = 0.0   # 禁用，与多原型头不兼容
        self.cutmix = 0.0  # 禁用
        self.smoothing = 0.1
        self.drop_path = 0.1
        
        # ===== MHF Enhancement 设置 =====
        self.use_enhanced_model = True
        self.enhancement_reduction = 0.0625
        
        # ===== 【V3 关键修复】学习率设置 =====
        # 目标: base_lr = 5e-4 (比 V2 提高 5 倍)
        # 原因: 1e-4 太低，CE Loss 不下降
        self.basic_lr_per_img = 5e-4 / 32  # 目标 LR = 5e-4 (batch=32)
        
        # ===== 【V3 关键修复】差异化学习率 =====
        # 提高 backbone LR，让特征能够适配医学图像
        self.backbone_lr_scale = 0.1   # 【修改】0.01 → 0.1 (提高 10 倍)
        self.mhf_lr_scale = 0.5        # 【修改】0.1 → 0.5 (提高 5 倍)
        self.enhancement_lr_scale = 1.0
        self.head_lr_scale = 2.0       # 【修改】5.0 → 2.0 (降低，因为 base LR 提高了)
        
        # ===== 不冻结 backbone =====
        self.freeze_backbone_epochs = 0
        
        # ===== 【V3 修复】Prototype Push 策略 =====
        # 更早开始 push，帮助原型快速锚定
        self.push_start_epoch = 10     # 【修改】30 → 10 (更早开始)
        self.push_interval = 5         # 【修改】10 → 5 (更频繁)
        self.push_end_epoch = max_epoch - 10
        self.push_momentum = 0.9       # 【修改】0.95 → 0.9 (更激进的更新)
        self.push_lr_recovery_epochs = 3
        self.push_lr_recovery_scale = 1.5
        
        # ===== 【V3 修复】学习率调度 =====
        self.warmup_epochs = 5         # 【修改】20 → 5 (更短 warmup)
        self.warmup_lr = 1e-5          # 【修改】1e-6 → 1e-5 (更高起始)
        self.min_lr = 1e-5             # 【修改】1e-6 → 1e-5 (更高最小 LR)
        self.cooldown_epochs = 10

        # ===== 其他设置 =====
        self.weights_prefix = ""
        self.save_folder_prefix = "ft_optimal_v3_"
        self.print_interval = 10
        self.dump_interval = 10
        self.eval_interval = 1


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 60)
    print("最优训练配置 V3 验证")
    print("=" * 60)
    
    # 验证学习率计算
    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"\n【学习率配置】")
    print(f"基础学习率: {exp.lr:.6f} (目标: 5e-4)")
    print(f"  - backbone: {exp.lr * exp.backbone_lr_scale:.6f} (目标: 5e-5)")
    print(f"  - mhf: {exp.lr * exp.mhf_lr_scale:.6f} (目标: 2.5e-4)")
    print(f"  - enhancement: {exp.lr * exp.enhancement_lr_scale:.6f} (目标: 5e-4)")
    print(f"  - head: {exp.lr * exp.head_lr_scale:.6f} (目标: 1e-3)")
    
    print(f"\n【Prototype Push 策略】")
    print(f"  - 开始 epoch: {exp.push_start_epoch}")
    print(f"  - 结束 epoch: {exp.push_end_epoch}")
    print(f"  - 间隔: {exp.push_interval}")
    print(f"  - 动量: {exp.push_momentum}")
    
    print(f"\n【数据增强】")
    print(f"  - Mixup: {exp.mixup} (禁用)")
    print(f"  - CutMix: {exp.cutmix} (禁用)")
    print(f"  - Label Smoothing: {exp.smoothing}")
    
    print(f"\n【Warmup 配置】")
    print(f"  - warmup_epochs: {exp.warmup_epochs}")
    print(f"  - warmup_lr: {exp.warmup_lr}")
    print(f"  - min_lr: {exp.min_lr}")
    
    print("\n" + "=" * 60)
    print("V3 vs V2 对比:")
    print("=" * 60)
    print("| 参数 | V2 | V3 | 变化 |")
    print("|------|----|----|------|")
    print(f"| base_lr | 1e-4 | 5e-4 | +5x |")
    print(f"| backbone_lr_scale | 0.01 | 0.1 | +10x |")
    print(f"| mhf_lr_scale | 0.1 | 0.5 | +5x |")
    print(f"| head_lr_scale | 5.0 | 2.0 | -2.5x |")
    print(f"| warmup_epochs | 20 | 5 | -4x |")
    print(f"| push_start_epoch | 30 | 10 | -20 |")
    print(f"| push_interval | 10 | 5 | -5 |")
