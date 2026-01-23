# MHF_block 增强模块使用手册

## 概述

MHF_block增强模块是对原始多层次特征融合模块(MHF_block)的非侵入式增强，通过添加轻量级注意力机制和动态缩放门控来提升模型性能。

### 核心特性

- **零初始化保证**: 增强模块初始输出为零，确保与预训练权重完全兼容
- **向后兼容**: 完全兼容原始MHF_block的接口和权重
- **轻量级设计**: 增强模块参数量 < 1%，几乎不增加计算开销
- **差异化学习率**: 支持backbone、MHF原始参数、增强模块三组不同学习率

---

## 快速开始

### 1. 基本使用

```python
# 使用增强版模型 (独立类，不影响原始main_model)
from projects.mae_lite.models_mae import main_model_enhanced

model = main_model_enhanced(
    num_classes=7,                    # 分类数
    enhancement_reduction=0.0625      # 通道缩减比例 (1/16)
)

# 或者使用注册的模型工厂函数
from projects.mae_lite.models_mae import HiFuse_Tiny_Enhanced
model = HiFuse_Tiny_Enhanced(num_classes=7, enhancement_reduction=0.0625)

# 注意: 预训练权重加载已集成在训练脚本中，通过 --resume 参数传入
# 训练脚本会自动使用 strict=False 加载权重，增强模块参数会自动跳过
```

### 1.1 Baseline对比实验

```python
# 原始模型 (完全不受增强功能影响)
from projects.mae_lite.models_mae import main_model

model_baseline = main_model(num_classes=7)

# 增强版模型
from projects.mae_lite.models_mae import main_model_enhanced

model_enhanced = main_model_enhanced(num_classes=7, enhancement_reduction=0.0625)

# 两者权重名称兼容 (fu1, fu2, fu3, fu4)，可以加载相同的预训练权重
```

### 2. 差异化学习率配置

```python
from projects.mae_lite.mhf_enhancement import get_parameter_groups

# 获取参数分组
param_groups = get_parameter_groups(
    model,
    backbone_lr_scale=0.1,      # backbone学习率 = base_lr * 0.1
    mhf_lr_scale=0.5,           # MHF原始参数学习率 = base_lr * 0.5
    enhancement_lr_scale=1.0,   # 增强模块学习率 = base_lr * 1.0
    base_lr=1e-4                # 基础学习率
)

# 创建优化器
optimizer = torch.optim.AdamW(param_groups)
```

### 3. 冻结/解冻控制

```python
# 冻结backbone (只训练MHF和增强模块)
model.freeze_backbone()

# 冻结MHF原始参数 (只训练增强模块)
model.freeze_mhf_original()

# 解冻所有参数
model.unfreeze_backbone()
```

---

## 详细配置

### MHFEnhancementConfig 配置类

```python
from projects.mae_lite.mhf_enhancement import MHFEnhancementConfig

# 创建配置
config = MHFEnhancementConfig(
    use_enhancement=True,           # 是否启用增强
    enhancement_reduction=0.0625,   # 通道缩减比例
    backbone_lr_scale=0.1,          # backbone学习率倍率
    mhf_lr_scale=0.5,               # MHF学习率倍率
    enhancement_lr_scale=1.0,       # 增强模块学习率倍率
    freeze_backbone_epochs=5,       # 冻结backbone的epoch数
    gradual_unfreeze=False,         # 是否渐进式解冻
    unfreeze_schedule=None          # 解冻计划
)

# 转换为字典
config_dict = config.to_dict()

# 从字典创建
config = MHFEnhancementConfig.from_dict(config_dict)
```

---

## 验证工具

### 1. 验证零初始化

```python
from projects.mae_lite.mhf_enhancement import verify_zero_init

# 验证增强模块的缩放因子是否为零
is_valid = verify_zero_init(model)
# 输出:
# Verifying zero initialization...
#   [OK] fu1.enhancement.scale: max value = 0.00e+00
#   [OK] fu2.enhancement.scale: max value = 0.00e+00
#   ...
# ✓ Zero initialization verified!
```

### 2. 验证权重加载

```python
from projects.mae_lite.mhf_enhancement import verify_weight_loading

# 验证预训练权重是否正确加载
is_valid = verify_weight_loading(model, 'pretrained_weights.pth')
# 输出:
# Verifying weight loading from: pretrained_weights.pth
#   Loaded: 245 keys
#   Skipped: 0 keys
#   Missing (new): 16 keys
#   Enhancement keys (expected missing): 16
# ✓ Weight loading verified!
```

### 3. 验证行为一致性

```python
from projects.mae_lite.mhf_enhancement import verify_behavior_consistency

# 比较原始模型和增强模型的输出差异
model_orig = main_model(num_classes=7, use_mhf_enhancement=False)
model_enhanced = main_model(num_classes=7, use_mhf_enhancement=True)

# 加载相同权重
model_orig.load_state_dict(checkpoint, strict=False)
model_enhanced.load_state_dict(checkpoint, strict=False)

is_consistent = verify_behavior_consistency(model_orig, model_enhanced)
# 输出:
# Verifying behavior consistency...
#   Max difference: 0.00e+00
#   Mean difference: 0.00e+00
#   Threshold: 1.00e-06
# ✓ Behavior consistency verified!
```

---

## 训练脚本集成

### 使用 finetuning_mae_exp.py

```bash
# 启用MHF增强训练 (使用 --use-enhanced-model 参数)
python projects/eval_tools/finetuning_mae_exp.py \
    --data-path /path/to/dataset \
    --num-classes 7 \
    --use-enhanced-model \
    --enhancement-reduction 0.0625 \
    --backbone-lr-scale 0.1 \
    --mhf-lr-scale 0.5 \
    --enhancement-lr-scale 1.0 \
    --epochs 100 \
    --batch-size 32 \
    --resume /path/to/pretrained_weights.pth
```

### Baseline对比实验

```bash
# 不启用增强 (原始main_model)
python projects/eval_tools/finetuning_mae_exp.py \
    --data-path /path/to/dataset \
    --num-classes 7 \
    --epochs 100 \
    --batch-size 32 \
    --resume /path/to/pretrained_weights.pth
    # 不加 --use-enhanced-model 参数，使用原始main_model
```

**注意**: 
- 不加 `--use-enhanced-model` 参数时使用原始 `main_model`
- 加 `--use-enhanced-model` 参数时使用增强版 `main_model_enhanced`
- 两者权重名称完全兼容，可以互相加载

---

## 学习率配置建议

针对 **ImageNet → 医学图像** 的迁移学习场景：

| 参数组 | 学习率倍率 | 说明 |
|--------|-----------|------|
| backbone (ConvNeXt + Swin) | 0.1x | 预训练特征，需要保护 |
| mhf_original (MHF_block原始部分) | 0.5x | 融合模块，需要适应新数据 |
| enhancement (新增增强模块) | 1.0x | 全新模块，需要快速学习 |

### 推荐训练策略

1. **阶段1 (Epoch 1-5)**: 冻结backbone，只训练MHF和增强模块
2. **阶段2 (Epoch 6-50)**: 解冻所有参数，使用差异化学习率
3. **阶段3 (Epoch 51-100)**: 降低学习率，精细调整

```python
# 阶段1: 冻结backbone
model.freeze_backbone()
for epoch in range(5):
    train_one_epoch(...)

# 阶段2: 解冻所有参数
model.unfreeze_backbone()
for epoch in range(5, 50):
    train_one_epoch(...)
```

---

## 模块结构

### EnhancementModuleLite (轻量版)

```
输入特征 (B, C, H, W)
    │
    ├── 通道注意力 (GAP + MLP + Sigmoid)
    │       │
    │       ▼
    │   通道加权特征
    │       │
    ├── 空间注意力 (MaxPool + AvgPool + Conv + Sigmoid)
    │       │
    │       ▼
    │   空间加权特征
    │       │
    └── 动态缩放 (scale * features, scale初始化为0)
            │
            ▼
        输出 (B, C, H, W)
```

### MHF_block_v2 结构

```
CNN特征 (l) ──┐
              │
Swin特征 (g) ─┼── 原始MHF_block逻辑 ── fuse
              │                          │
前层特征 (f) ─┘                          │
                                         │
                    ┌────────────────────┘
                    │
                    ▼
              [use_enhancement=True?]
                    │
            ┌───────┴───────┐
            │               │
           Yes             No
            │               │
            ▼               │
    EnhancementModuleLite   │
            │               │
            ▼               │
    fuse + enhancement      │
            │               │
            └───────┬───────┘
                    │
                    ▼
                输出特征
```

---

## 常见问题

### Q1: 为什么增强模块初始输出为零？

**A**: 这是零初始化设计的核心。通过将缩放因子`scale`初始化为0，确保：
- 加载预训练权重后，模型行为与原始模型完全一致
- 训练过程中，增强模块逐渐学习到合适的缩放值
- 避免随机初始化破坏预训练特征

### Q2: 如何选择 enhancement_reduction 参数？

**A**: 默认值 `0.0625` (1/16) 适用于大多数场景。如果：
- 数据集较小：可以增大到 `0.125` (1/8) 减少参数
- 数据集较大：可以减小到 `0.03125` (1/32) 增加容量

### Q3: 是否需要实现自定义权重加载函数？

**A**: 不需要。训练脚本已经集成了权重加载功能：
- 通过 `--resume` 参数传入预训练权重路径
- 脚本内部使用 `strict=False` 自动处理缺失的enhancement参数
- 无需手动调用 `torch.load()` 和 `load_state_dict()`

### Q4: 如何只训练增强模块？

**A**: 
```python
# 冻结backbone
model.freeze_backbone()
# 冻结MHF原始参数
model.freeze_mhf_original()
# 此时只有enhancement模块参与训练
```

### Q5: 如何跑baseline对比实验？

**A**: 使用不同的模型类即可：
```python
# Baseline (原始main_model)
from projects.mae_lite.models_mae import main_model
model_baseline = main_model(num_classes=7)

# 增强版 (main_model_enhanced)
from projects.mae_lite.models_mae import main_model_enhanced
model_enhanced = main_model_enhanced(num_classes=7, enhancement_reduction=0.0625)
```

或者在训练脚本中：
```bash
# Baseline
python projects/eval_tools/finetuning_mae_exp.py \
    --data-path /path/to/dataset \
    --num-classes 7 \
    --resume /path/to/weights.pth

# 增强版 (加 --use-enhanced-model)
python projects/eval_tools/finetuning_mae_exp.py \
    --data-path /path/to/dataset \
    --num-classes 7 \
    --use-enhanced-model \
    --resume /path/to/weights.pth
```
两者使用相同的预训练权重，权重名称完全兼容（fu1, fu2, fu3, fu4）。

---

## 文件结构

```
projects/mae_lite/
├── mhf_enhancement.py      # 增强模块核心实现
│   ├── EnhancementModuleLite   # 轻量版增强模块
│   ├── AMAFDSGModule           # 完整版增强模块
│   ├── MHF_block_v2            # 增强版MHF_block
│   ├── get_parameter_groups    # 参数分组函数
│   ├── verify_zero_init        # 零初始化验证
│   ├── verify_weight_loading   # 权重加载验证
│   ├── verify_behavior_consistency  # 行为一致性验证
│   └── MHFEnhancementConfig    # 配置数据类
│
├── models_mae.py           # 主模型文件
│   ├── main_model              # 原始模型 (不受增强功能影响)
│   ├── main_model_enhanced     # 增强版模型 (继承自main_model)
│   ├── HiFuse_Tiny             # 原始HiFuse_Tiny
│   ├── HiFuse_Small            # 原始HiFuse_Small
│   ├── HiFuse_Base             # 原始HiFuse_Base
│   ├── HiFuse_Tiny_Enhanced    # 增强版HiFuse_Tiny
│   ├── HiFuse_Small_Enhanced   # 增强版HiFuse_Small
│   └── HiFuse_Base_Enhanced    # 增强版HiFuse_Base
│
└── MHF_ENHANCEMENT_使用手册.md  # 本文档
```

### 模型选择

| 模型类 | 说明 | 使用场景 |
|--------|------|----------|
| `main_model` | 原始模型，使用MHF_block | Baseline实验、对比实验 |
| `main_model_enhanced` | 增强版模型，使用MHF_block_v2 | 增强实验、迁移学习 |

---

## 版本信息

- **版本**: 1.0.0
- **作者**: Kiro AI Assistant
- **日期**: 2026-01-21
- **兼容性**: PyTorch >= 1.9.0

