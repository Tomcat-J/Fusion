# --------------------------------------------------------
# 渐进式微调配置 v4 - 禁用 Push + 优化学习率调度
# 
# 【v3 问题诊断】
# 1. Best 88.462% 在 epoch 43，之后未能超越
# 2. Prototype Push 导致准确率波动，但未带来提升
# 3. 学习率在后期可能过低，限制了进一步优化
# 
# 【v4 核心改进】
# 1. 禁用 Prototype Push（已证明对当前任务无益）
# 2. 使用更长的 warmup 和更平缓的学习率衰减
# 3. 启用轻度 mixup/cutmix 增强泛化
# 4. 调整损失权重，增强类间分离
# --------------------------------------------------------
import os
import torch
from loguru import logger
from projects.eval_tools.finetuning_stage2_exp import Exp as BaseExp


class Exp(BaseExp):
    """
    渐进式微调 v4：禁用 Push + 优化学习率
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

        # ===== 数据增强 - 启用轻度 mixup =====
        self.color_jitter = 0.4
        self.aa = "rand-m9-mstd0.5-inc1"
        self.reprob = 0.25
        self.remode = "pixel"
        self.recount = 1
        self.mixup = 0.1              # 启用轻度 mixup
        self.cutmix = 0.1             # 启用轻度 cutmix
        self.mixup_prob = 0.5         # 50% 概率应用
        self.smoothing = 0.1
        self.drop_path = 0.15         # 略微增加 drop_path
        
        # ===== 学习率设置 =====
        self.basic_lr_per_img = 1e-4 / 32
        
        # ===== 分层学习率 =====
        self.backbone_lr_scale = 0.05
        self.mhf_lr_scale = 0.2
        self.enhancement_lr_scale = 3.0
        self.head_lr_scale = 2.0
        
        # ===== 学习率调度 - 更平缓的衰减 =====
        self.sched = "warmcos"
        self.warmup_epochs = 5         # 延长 warmup
        self.warmup_lr = 1e-6
        self.min_lr = 5e-7             # 更低的最小学习率
        self.cooldown_epochs = 10      # 延长 cooldown

        # ===== 冻结策略 =====
        self.freeze_backbone_epochs = 3

        # ===== 【关键】禁用 Prototype Push =====
        self.push_start_epoch = 999    # 设置为超大值，实际禁用
        self.push_interval = 999
        self.push_end_epoch = 0
        self.push_momentum = 0.0

        # ===== EMA =====
        self.model_ema = True
        self.model_ema_decay = 0.9998
        self.model_ema_force_cpu = False

        # ===== 路径设置 =====
        self.weights_prefix = ""
        self.pretrain_exp_name = "HiFuse_Small_1e-5-0.05"
        self.save_folder_prefix = "ft_progressive_v4_"
        self.print_interval = 10
        self.dump_interval = 5
        self.eval_interval = 1
        
        # ===== 损失函数权重 - 增强类间分离 =====
        self.loss_weights = {
            'clst_scale': 0.8,
            'sep_scale': 0.25,         # 进一步增强类间分离
            'orth_scale': 0.02,
            'local_consistency_scale': 0.1,
            'margin': 0.5,             # 增大 margin
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
            
            logger.info(f"[Exp v4] Loss weights adjusted:")
            logger.info(f"  sep_scale: 0.12 -> {loss_fn.sep_scale}")
            logger.info(f"  orth_scale: 0.08 -> {loss_fn.orth_scale}")
            logger.info(f"  local_consistency_scale: 0.05 -> {loss_fn.local_consistency_scale}")
            logger.info(f"  margin: 0.3 -> {loss_fn.margin}")
        
        return model


class ExpNoPush(Exp):
    """
    纯净版：禁用 Push，禁用 mixup（对比实验）
    """
    def __init__(self, batch_size, max_epoch=100):
        super(ExpNoPush, self).__init__(batch_size, max_epoch)
        self.mixup = 0.0
        self.cutmix = 0.0
        self.drop_path = 0.1
        self.save_folder_prefix = "ft_progressive_v4_nopush_nomix_"


class ExpHigherLR(Exp):
    """
    更高学习率版本（如果 v4 效果不够好）
    """
    def __init__(self, batch_size, max_epoch=100):
        super(ExpHigherLR, self).__init__(batch_size, max_epoch)
        
        self.basic_lr_per_img = 1.5e-4 / 32  # 提高 50%
        self.backbone_lr_scale = 0.08
        self.mhf_lr_scale = 0.3
        self.head_lr_scale = 2.5
        
        self.save_folder_prefix = "ft_progressive_v4_higherlr_"


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 70)
    print("渐进式微调配置 v4 - 禁用 Push + 优化学习率")
    print("=" * 70)
    
    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"\n【学习率配置】")
    print(f"基础学习率: {exp.lr:.6f}")
    print(f"backbone LR: {exp.lr * exp.backbone_lr_scale:.6f} ({exp.backbone_lr_scale}x)")
    print(f"MHF LR: {exp.lr * exp.mhf_lr_scale:.6f} ({exp.mhf_lr_scale}x)")
    print(f"enhancement LR: {exp.lr * exp.enhancement_lr_scale:.6f} ({exp.enhancement_lr_scale}x)")
    print(f"head LR: {exp.lr * exp.head_lr_scale:.6f} ({exp.head_lr_scale}x)")
    
    print(f"\n【v4 vs v3 关键改进】")
    print("1. ✅ 禁用 Prototype Push（v3 中 push 导致波动但无提升）")
    print("2. ✅ 启用轻度 mixup/cutmix (0.1) 增强泛化")
    print("3. ✅ 延长 warmup: 3 -> 5 epochs")
    print("4. ✅ 延长 cooldown: 5 -> 10 epochs")
    print("5. ✅ 增大 sep_scale: 0.2 -> 0.25")
    print("6. ✅ 增大 margin: 0.4 -> 0.5")
    
    print(f"\n【损失权重】")
    for k, v in exp.loss_weights.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("使用方法:")
    print("python mae_lite/tools/train.py \\")
    print("  --exp_file projects/eval_tools/finetuning_progressive_v4_exp.py \\")
    print("  --ckpt outputs/HiFuse_Small_1e-5-0.05/ft_transfer_eval/867last_epoch_best_ckpt.checkpoints.tar \\")
    print("  --devices 0 --batch_size 32 --eval")
    print("=" * 70)
