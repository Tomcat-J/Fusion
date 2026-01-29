# --------------------------------------------------------
# 渐进式微调配置 v3p - 稳定 Push + 适度分离增强
#
# 【设计目标】
# - 保留 v3 的后期提升潜力（观察到晚期仍能创新高）
# - 降低 Push 频率与动量，减少精度大幅波动
# - 采用更平滑的 warmup/cooldown 以提升稳定性
# --------------------------------------------------------
import os
import torch
from loguru import logger
from projects.eval_tools.finetuning_stage2_exp import Exp as BaseExp


class Exp(BaseExp):
    """
    渐进式微调 v3p：轻量 Push + 稳定调度
    """
    def __init__(self, batch_size, max_epoch=100):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 使用增强版模型 =====
        self.use_enhanced_model = True
        self.enhancement_reduction = 0.0625

        # ===== 优化器设置 =====
        self.opt = "adamw"
        self.weight_decay = 0.05
        self.clip_grad = 1.0
        self.clip_mode = "norm"

        # ===== 数据增强（与 v3 一致，避免过强正则） =====
        self.color_jitter = 0.4
        self.aa = "rand-m9-mstd0.5-inc1"
        self.reprob = 0.25
        self.remode = "pixel"
        self.recount = 1
        self.mixup = 0.0
        self.cutmix = 0.0
        self.smoothing = 0.1
        self.drop_path = 0.1

        # ===== 学习率设置 =====
        self.basic_lr_per_img = 1e-4 / 32

        # ===== 分层学习率 =====
        self.backbone_lr_scale = 0.05
        self.mhf_lr_scale = 0.2
        self.enhancement_lr_scale = 3.0
        self.head_lr_scale = 2.0

        # ===== 学习率调度 - 更平缓的衰减 =====
        self.sched = "warmcos"
        self.warmup_epochs = 5
        self.warmup_lr = 1e-6
        self.min_lr = 7.5e-7
        self.cooldown_epochs = 10

        # ===== 冻结策略 =====
        self.freeze_backbone_epochs = 3

        # ===== Prototype Push（延后 + 降频 + 降动量） =====
        self.push_start_epoch = 24
        self.push_interval = 12
        self.push_end_epoch = 84
        self.push_momentum = 0.75

        # ===== EMA =====
        self.model_ema = True
        self.model_ema_decay = 0.9998
        self.model_ema_force_cpu = False

        # ===== 路径设置 =====
        self.weights_prefix = ""
        self.pretrain_exp_name = "HiFuse_Small_1e-5-0.05"
        self.save_folder_prefix = "ft_progressive_v3p_"
        self.print_interval = 10
        self.dump_interval = 5
        self.eval_interval = 1

        # ===== 损失函数权重 - 介于 v3/v4 之间 =====
        self.loss_weights = {
            'clst_scale': 0.8,
            'sep_scale': 0.22,
            'orth_scale': 0.02,
            'local_consistency_scale': 0.1,
            'margin': 0.45,
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

            logger.info(f"[Exp v3p] Loss weights adjusted:")
            logger.info(f"  sep_scale: 0.12 -> {loss_fn.sep_scale}")
            logger.info(f"  orth_scale: 0.08 -> {loss_fn.orth_scale}")
            logger.info(f"  local_consistency_scale: 0.05 -> {loss_fn.local_consistency_scale}")
            logger.info(f"  margin: 0.3 -> {loss_fn.margin}")

        return model


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 70)
    print("渐进式微调配置 v3p - 稳定 Push + 平滑调度")
    print("=" * 70)

    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"\n【学习率配置】")
    print(f"基础学习率: {exp.lr:.6f}")
    print(f"backbone LR: {exp.lr * exp.backbone_lr_scale:.6f} ({exp.backbone_lr_scale}x)")
    print(f"MHF LR: {exp.lr * exp.mhf_lr_scale:.6f} ({exp.mhf_lr_scale}x)")
    print(f"enhancement LR: {exp.lr * exp.enhancement_lr_scale:.6f} ({exp.enhancement_lr_scale}x)")
    print(f"head LR: {exp.lr * exp.head_lr_scale:.6f} ({exp.head_lr_scale}x)")

    print(f"\n【v3p 关键改进】")
    print("1. ✅ Push 延后到 epoch 24，降低早期扰动")
    print("2. ✅ Push 频率降低到 12 epochs")
    print("3. ✅ Push 动量降到 0.75，减少剧烈跳变")
    print("4. ✅ warmup/cooldown 更平滑，降低后期抖动")
    print("5. ✅ sep/margin 介于 v3/v4 之间，兼顾分离与稳定")

    print(f"\n【损失权重】")
    for k, v in exp.loss_weights.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("使用方法:")
    print("python mae_lite/tools/train.py \\")
    print("  --exp_file projects/eval_tools/finetuning_progressive_v3p_exp.py \\")
    print("  --ckpt outputs/HiFuse_Small_1e-5-0.05/ft_progressive_v3_eval/88.4last_epoch_best_ckpt.checkpoints.tar \\")
    print("  --devices 0 --batch_size 32 --eval")
    print("=" * 70)
