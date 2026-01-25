# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# 权重迁移微调配置
# 
# 【场景】
# 源模型: 原始 Head (head.py) + 原始 backbone (无增强模块)
# 目标模型: HeadOptimal (多原型) + 增强版 backbone (MHF_block_v2 + EnhancementModuleLite)
# 
# 【策略】
# 1. 加载源模型的 backbone 权重 (完全匹配)
# 2. 加载源模型的 MHF_block 权重 (完全匹配)
# 3. 迁移 Head 权重:
#    - cross_attn, self_attn, proj: 直接迁移
#    - class_embeddings → prototypes: 用类别嵌入初始化每类第一个原型
# 4. 新增模块随机初始化:
#    - EnhancementModuleLite
#    - mixed_attn.light_attn, gate
#    - ffn
# 5. 差异化学习率:
#    - backbone: 0.01x (几乎冻结)
#    - MHF: 0.1x (低学习率)
#    - head_transferred: 0.5x (已迁移部分)
#    - head_new: 5.0x (新增部分)
#    - enhancement: 5.0x (新增模块)
# --------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Dict, List, Tuple, Optional
from projects.eval_tools.finetuning_mae_exp import Exp as BaseExp


class Exp(BaseExp):
    """
    权重迁移微调配置
    
    从原始 Head + backbone 迁移到 HeadOptimal + 增强版 backbone
    """
    def __init__(self, batch_size, max_epoch=100):
        super(Exp, self).__init__(batch_size, max_epoch)

        # ===== 【关键】使用增强版模型 =====
        self.use_enhanced_model = True
        self.enhancement_reduction = 0.0625
        
        # ===== 优化器设置 =====
        self.opt = "adamw"
        self.weight_decay = 0.05
        self.clip_grad = 1.0
        self.clip_mode = "norm"

        # ===== 数据增强 =====
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
        
        # ===== 差异化学习率 =====
        self.backbone_lr_scale = 0.01    # backbone 几乎冻结
        self.mhf_lr_scale = 0.1          # MHF 低学习率
        self.enhancement_lr_scale = 5.0  # 增强模块高学习率
        self.head_lr_scale = 1.0         # head 整体学习率
        
        # ===== 学习率调度 =====
        self.sched = "warmcos_scale"
        self.warmup_epochs = 5
        self.warmup_lr = 1e-7
        self.min_lr = 1e-7
        self.cooldown_epochs = 10

        # ===== 冻结策略 =====
        self.freeze_backbone_epochs = 10  # 前 10 epochs 冻结 backbone
        
        # ===== Prototype Push 策略 =====
        self.push_start_epoch = 15
        self.push_interval = 10
        self.push_end_epoch = 90
        self.push_momentum = 0.9

        # ===== EMA =====
        self.model_ema = True
        self.model_ema_decay = 0.9998
        self.model_ema_force_cpu = False

        # ===== 路径设置 =====
        self.weights_prefix = ""
        self.pretrain_exp_name = "HiFuse_Small_1e-5-0.05"
        self.save_folder_prefix = "ft_transfer_"
        self.print_interval = 10
        self.dump_interval = 10
        self.eval_interval = 5

    def set_current_state(self, current_step, ckpt_path=None):
        """
        重写权重加载逻辑，支持从原始 Head 迁移到 HeadOptimal
        """
        if current_step == 0:
            if ckpt_path is None:
                # 默认使用 83% 准确率的最佳权重
                ckpt_path = os.path.join(
                    self.output_dir, 
                    self.pretrain_exp_name, 
                    "ft_eval",
                    "last_epoch_best_ckpt.checkpoints.tar"
                )
            
            logger.info(f"Loading source weights from: {ckpt_path}")
            msg = self.transfer_weights(ckpt_path, map_location="cpu")
            logger.info(f"Transfer complete. Missing keys: {len(msg.missing_keys)}")

    def transfer_weights(self, ckpt_path: str, map_location: str = "cpu"):
        """
        权重迁移：从原始模型迁移到增强版模型
        
        迁移策略:
        1. backbone: 直接迁移 (形状完全匹配)
        2. MHF_block: 直接迁移原始部分，enhancement 随机初始化
        3. Head: 
           - proj, cross_attn, self_attn: 迁移
           - class_embeddings → prototypes: 特殊处理
           - 新增模块: 随机初始化
        """
        from torch.nn.modules.module import _IncompatibleKeys
        
        if not os.path.isfile(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
        
        ckpt = torch.load(ckpt_path, map_location=map_location)
        
        # 获取源 state_dict
        if "model" in ckpt:
            src_state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            src_state_dict = ckpt["state_dict"]
        else:
            src_state_dict = ckpt
        
        # 获取目标模型
        model = self.get_model()
        tgt_state_dict = model.state_dict()
        
        # 构建新的 state_dict
        new_state_dict = {}
        transferred_keys = []
        skipped_keys = []
        special_keys = []
        
        # 定义 Head 参数映射
        head_mapping = {
            # 原始 Head → HeadOptimal
            'head.proj.weight': 'head.proj.0.weight',
            'head.proj.bias': 'head.proj.0.bias',
            'head.cross_attn.in_proj_weight': 'head.mixed_attn.cross_attn.in_proj_weight',
            'head.cross_attn.in_proj_bias': 'head.mixed_attn.cross_attn.in_proj_bias',
            'head.cross_attn.out_proj.weight': 'head.mixed_attn.cross_attn.out_proj.weight',
            'head.cross_attn.out_proj.bias': 'head.mixed_attn.cross_attn.out_proj.bias',
            'head.self_attn.in_proj_weight': 'head.mixed_attn.self_attn.in_proj_weight',
            'head.self_attn.in_proj_bias': 'head.mixed_attn.self_attn.in_proj_bias',
            'head.self_attn.out_proj.weight': 'head.mixed_attn.self_attn.out_proj.weight',
            'head.self_attn.out_proj.bias': 'head.mixed_attn.self_attn.out_proj.bias',
        }
        
        for src_key, src_value in src_state_dict.items():
            # 添加 "model." 前缀
            if not src_key.startswith("model."):
                src_key_full = "model." + src_key
            else:
                src_key_full = src_key
                src_key = src_key.replace("model.", "")
            
            # 检查是否需要特殊映射 (Head 参数)
            if src_key in head_mapping:
                tgt_key = "model." + head_mapping[src_key]
                if tgt_key in tgt_state_dict:
                    if src_value.shape == tgt_state_dict[tgt_key].shape:
                        new_state_dict[tgt_key] = src_value
                        transferred_keys.append(f"{src_key} → {head_mapping[src_key]}")
                    else:
                        skipped_keys.append(f"{src_key} (shape mismatch)")
                continue
            
            # 特殊处理: class_embeddings → prototypes
            if 'class_embeddings' in src_key:
                tgt_key = "model.head.proto_manager.prototypes"
                if tgt_key in tgt_state_dict:
                    # 用类别嵌入初始化每类的第一个原型
                    self._init_prototypes_from_embeddings(
                        model, src_value, tgt_state_dict[tgt_key]
                    )
                    special_keys.append(f"{src_key} → prototypes (special init)")
                continue
            
            # 直接迁移 (backbone, MHF 等)
            if src_key_full in tgt_state_dict:
                if src_value.shape == tgt_state_dict[src_key_full].shape:
                    new_state_dict[src_key_full] = src_value
                    transferred_keys.append(src_key)
                else:
                    skipped_keys.append(f"{src_key} (shape: {src_value.shape} vs {tgt_state_dict[src_key_full].shape})")
            else:
                skipped_keys.append(f"{src_key} (not in target)")
        
        # 加载权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        
        # 打印统计
        logger.info(f"=== Weight Transfer Summary ===")
        logger.info(f"Transferred: {len(transferred_keys)} parameters")
        logger.info(f"Skipped: {len(skipped_keys)} parameters")
        logger.info(f"Special: {len(special_keys)} parameters")
        logger.info(f"Missing (new modules): {len(msg.missing_keys)} parameters")
        
        if special_keys:
            logger.info(f"Special transfers: {special_keys}")
        
        # 打印新增模块 (随机初始化)
        new_modules = [k for k in msg.missing_keys if any(
            pattern in k for pattern in ['enhancement', 'light_attn', 'gate', 'ffn', 'proto_manager']
        )]
        if new_modules:
            logger.info(f"New modules (random init): {len(new_modules)} parameters")
            logger.info(f"  Examples: {new_modules[:5]}...")
        
        return msg
    
    def _init_prototypes_from_embeddings(
        self, 
        model, 
        class_embeddings: torch.Tensor,
        target_prototypes: torch.Tensor
    ):
        """
        用类别嵌入初始化原型
        
        策略:
        1. 每类的第一个原型 = 类别嵌入
        2. 其余原型 = 类别嵌入 + 小扰动
        
        Args:
            model: 目标模型
            class_embeddings: 原始类别嵌入 (num_classes, emb_dim)
            target_prototypes: 目标原型张量 (num_classes * max_prototypes, emb_dim)
        """
        num_classes = class_embeddings.size(0)
        emb_dim = class_embeddings.size(1)
        max_prototypes = target_prototypes.size(0) // num_classes
        
        logger.info(f"Initializing prototypes from class embeddings:")
        logger.info(f"  Source: {class_embeddings.shape}")
        logger.info(f"  Target: {target_prototypes.shape}")
        logger.info(f"  num_classes={num_classes}, max_prototypes={max_prototypes}")
        
        # 获取 head 的 proto_manager
        head = model.model.head
        if hasattr(head, 'proto_manager'):
            proto_manager = head.proto_manager
            
            with torch.no_grad():
                for c in range(num_classes):
                    # 归一化类别嵌入
                    class_emb = F.normalize(class_embeddings[c], p=2, dim=-1)
                    
                    # 获取该类的原型数量
                    num_protos = proto_manager.protos_per_class[c].item()
                    
                    for p in range(num_protos):
                        proto_idx = c * max_prototypes + p
                        
                        if p == 0:
                            # 第一个原型 = 类别嵌入
                            proto_manager.prototypes.data[proto_idx] = class_emb
                        else:
                            # 其余原型 = 类别嵌入 + 小扰动
                            noise = torch.randn_like(class_emb) * 0.1
                            proto_manager.prototypes.data[proto_idx] = F.normalize(
                                class_emb + noise, p=2, dim=-1
                            )
            
            logger.info(f"  Initialized {num_classes * num_protos} prototypes")
        else:
            logger.warning("  proto_manager not found, skipping prototype initialization")


class ExpConservative(Exp):
    """
    保守迁移配置 - 更低的学习率，更长的冻结期
    """
    def __init__(self, batch_size, max_epoch=100):
        super(ExpConservative, self).__init__(batch_size, max_epoch)
        
        # 更保守的学习率
        self.backbone_lr_scale = 0.001   # 几乎完全冻结
        self.mhf_lr_scale = 0.05         # 非常低
        self.enhancement_lr_scale = 3.0  # 中等
        self.head_lr_scale = 0.5         # 较低
        
        # 更长的冻结期
        self.freeze_backbone_epochs = 20
        
        # 更晚开始 push
        self.push_start_epoch = 30
        
        self.save_folder_prefix = "ft_transfer_conservative_"


if __name__ == "__main__":
    exp = Exp(32)
    print("=" * 70)
    print("权重迁移微调配置")
    print("=" * 70)
    
    exp.lr = exp.basic_lr_per_img * exp.batch_size
    print(f"\n【学习率配置】")
    print(f"基础学习率: {exp.lr:.6f}")
    print(f"backbone LR: {exp.lr * exp.backbone_lr_scale:.8f} ({exp.backbone_lr_scale}x)")
    print(f"MHF LR: {exp.lr * exp.mhf_lr_scale:.6f} ({exp.mhf_lr_scale}x)")
    print(f"enhancement LR: {exp.lr * exp.enhancement_lr_scale:.6f} ({exp.enhancement_lr_scale}x)")
    print(f"head LR: {exp.lr * exp.head_lr_scale:.6f} ({exp.head_lr_scale}x)")
    
    print(f"\n【迁移策略】")
    print("源模型: 原始 Head + 原始 backbone")
    print("目标模型: HeadOptimal + 增强版 backbone")
    print("")
    print("迁移内容:")
    print("  ✅ backbone: 直接迁移")
    print("  ✅ MHF_block: 直接迁移原始部分")
    print("  ✅ Head.proj: 迁移到 HeadOptimal.proj")
    print("  ✅ Head.cross_attn: 迁移到 mixed_attn.cross_attn")
    print("  ✅ Head.self_attn: 迁移到 mixed_attn.self_attn")
    print("  ⚠️ class_embeddings → prototypes: 特殊初始化")
    print("  ❌ EnhancementModuleLite: 随机初始化")
    print("  ❌ mixed_attn.light_attn, gate: 随机初始化")
    print("  ❌ ffn: 随机初始化")
    
    print("\n" + "=" * 70)
    print("使用方法:")
    print("python mae_lite/tools/train.py \\")
    print("  --exp_file projects/eval_tools/finetuning_transfer_exp.py \\")
    print("  --ckpt outputs/HiFuse_Small_1e-5-0.05/ft_eval/last_epoch_best_ckpt.checkpoints.tar \\")
    print("  --devices 0 --batch_size 32 --eval")
    print("=" * 70)
