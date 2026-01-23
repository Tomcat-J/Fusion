# 局部特征 Push 增强 - 使用指南

## 概述

本增强将原型 Push 机制从**全局池化特征**改为**局部 patch 特征**，使原型能够锚定到真实的局部图像区域。

## 核心改进

### 原方法 vs 新方法

| 特性 | 原方法 (全局) | 新方法 (局部) |
|------|--------------|--------------|
| 使用特征 | `x_fu` (B, 1, 768) | `x_fu1` (B, 49, 768) |
| 特征类型 | 全局平均池化 | 7×7 局部 patches |
| 原型特点 | 通用、平滑 | 细节、多样 |
| 理论支持 | - | ProtoPNet, ProtoASNet, GPA |

### 理论依据

1. **ProtoPNet (NIPS 2019)**: 原型应锚定到局部 patch，而非全局特征
2. **ProtoASNet (CVPR 2023)**: 局部对齐优于全局池化
3. **GPA (2025)**: Local alignment 显著提升原型多样性

## 使用方法

### 默认使用 (推荐)

```python
# 训练脚本中，push_prototypes 默认使用局部模式
model.push_prototypes(train_loader, device)  # use_local=True (默认)
```

### 显式指定模式

```python
# 使用局部 patch 特征 (推荐)
model.push_prototypes(train_loader, device, use_local=True, momentum=0.9)

# 使用全局池化特征 (备选)
model.push_prototypes(train_loader, device, use_local=False)
```

### 直接调用局部 Push

```python
# 直接调用局部 Push 方法
model.push_prototypes_local(
    dataloader=train_loader,
    device=device,
    momentum=0.9,              # 动量系数
    max_patches_per_class=5000  # 每类最大 patch 数
)
```

## 配置参数

### push_prototypes 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataloader` | DataLoader | 必需 | 训练数据加载器 |
| `device` | str/device | 必需 | GPU 设备 |
| `use_local` | bool | True | 是否使用局部 patch 特征 |
| `momentum` | float | 0.9 | 动量系数 |

### push_prototypes_local 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataloader` | DataLoader | 必需 | 训练数据加载器 |
| `device` | str/device | 必需 | GPU 设备 |
| `momentum` | float | 0.9 | 动量系数 (高动量防漂移) |
| `max_patches_per_class` | int | 5000 | 每类最大收集 patch 数 |

## 训练脚本配置

在 `train.py` 中的 Push 调用无需修改，默认使用局部模式：

```python
# train.py 中的现有代码
if hasattr(inner_model, 'push_prototypes') and (epoch + 1) >= push_start_epoch and (epoch + 1) % push_interval == 0:
    inner_model.push_prototypes(train_loader, gpu)  # 自动使用局部模式
```

如需切换到全局模式：

```python
inner_model.push_prototypes(train_loader, gpu, use_local=False)
```

## 算法流程

```
对于每个类别 c:
    1. 收集所有同类样本的局部 patch 特征
       - 每个样本有 49 个 patches (7×7)
       - 总共 N_c × 49 个 patches
    
    2. 对于每个原型 p_k:
        a. 计算 p_k 与所有同类 patches 的 L2 距离
        b. 找到最近的 patch 特征 f_nearest
        c. 动量更新: p_k = 0.9 * p_k + 0.1 * f_nearest
        d. L2 归一化: p_k = p_k / ||p_k||
```

## 内存优化

为防止内存溢出，实现了以下优化：

1. **分批收集**: 逐批处理数据，不一次性加载
2. **数量限制**: 每类最多收集 5000 个 patches
3. **CPU 缓存**: 收集的特征先存 CPU，更新时再移到 GPU
4. **提前终止**: 所有类别收集够后停止遍历

## 输出日志示例

```
Collecting LOCAL patch features for prototype push...
  Using x_fu1 (B, 49, 768) instead of x_fu (B, 1, 768)
  Processed 50 batches...
  Processed 100 batches...
Patch collection statistics:
  Class 0: 4900 patches
  Class 1: 2450 patches
  Class 2: 1470 patches
  ...
Updating prototypes using LOCAL patches...
  Class 0: 4900 patches
    Updated 5 prototypes, min_dist: 0.1234
  Class 1: 2450 patches
    Updated 5 prototypes, min_dist: 0.1567
  ...
Local prototype push completed!
```

## 与 head_optimal.py 的兼容性

局部 Push 完全兼容 `head_optimal.py`:

- 使用 `head.proj` 投影特征到嵌入空间
- 使用 `head.proto_manager` 管理原型
- 直接更新 `proto_manager.prototypes.data`

## 注意事项

1. **动量选择**: 推荐使用高动量 (0.9)，防止原型漂移过快
2. **Push 时机**: 建议在 epoch 10+ 开始 Push，让模型先稳定
3. **Push 频率**: 建议每 5 个 epoch Push 一次
4. **内存监控**: 如遇内存问题，减小 `max_patches_per_class`

## 常见问题

### Q: 为什么推荐使用局部特征？

A: 全局池化会平均掉局部细节，导致原型过于通用。局部 patch 特征保留了空间变异性，使原型能够捕捉子模式（如皮肤病变的边缘、纹理等）。

### Q: 动量为什么设为 0.9？

A: 高动量确保原型更新平滑，避免因单个异常 patch 导致原型突变。0.9 意味着每次更新只有 10% 来自新特征。

### Q: 如何验证 Push 效果？

A: 观察日志中的 `min_dist` 值，应该随训练逐渐减小，表示原型越来越接近真实特征。
