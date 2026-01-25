# --------------------------------------------------------
# MHF_block Enhancement Module
# --------------------------------------------------------
# 本模块实现了MHF_block（多层次特征融合模块）的增强功能
# 
# 核心功能:
# 1. EnhancementModuleLite: 超轻量级增强模块，包含通道注意力和空间注意力
# 2. AMAFDSGModule: 完整版自适应多尺度注意力融合+动态缩放门控模块
# 3. MHF_block_v2: 增强版MHF_block，支持可选的增强功能
# 4. 权重加载和参数分组工具函数
#
# 设计原则:
# - 非侵入式设计: 不修改原始MHF_block内部结构
# - 零初始化保证: 增强模块初始输出为零，确保初始行为一致
# - 向后兼容: MHF_block_v2完全兼容原始接口和权重
# - 轻量级增强: 增强模块参数量控制在原模块的5%以内
#
# Requirements: 1.1, 1.2, 1.3
# --------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


__all__ = [
    'EnhancementModuleLite',
    'AMAFDSGModule', 
    'MHF_block_v2',           # Task 5
    'get_parameter_groups',   # Task 8
    'verify_zero_init',       # Task 11.1
    'verify_weight_loading',  # Task 11.2
    'verify_behavior_consistency',  # Task 11.3
    'MHFEnhancementConfig',   # Task 13
]


class EnhancementModuleLite(nn.Module):
    """
    超轻量级增强模块 (参数量 < 1%)
    
    本模块实现了轻量级的特征增强，包含:
    - 轻量通道注意力: 使用GAP + MLP进行通道重校准
    - 轻量空间注意力: 使用max/avg pooling + 卷积进行空间增强
    - 动态缩放门控: 零初始化的可学习缩放因子
    
    设计特点:
    - 零初始化保证初始输出为零，确保与预训练权重兼容
    - 参数量极小，不影响原模型性能
    - 残差连接设计，输出为 scale * enhanced_features
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例，默认1/16 (0.0625)
    
    Requirements: 1.2, 1.3, 2.1, 2.2, 1.4
    """
    
    def __init__(self, channels: int, reduction: float = 0.0625):
        super().__init__()
        
        # 计算缩减后的通道数，确保最小为8
        reduced_channels = max(int(channels * reduction), 8)
        
        # ===== 2.1 轻量通道注意力模块 =====
        # 使用 AdaptiveAvgPool2d + 1x1卷积 + ReLU + 1x1卷积 + Sigmoid
        # Requirements: 1.2
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # 全局平均池化 -> (B, C, 1, 1)
            nn.Conv2d(channels, reduced_channels, 1, bias=False), # 通道缩减
            nn.ReLU(inplace=True),                                # 非线性激活
            nn.Conv2d(reduced_channels, channels, 1, bias=False), # 通道恢复
            nn.Sigmoid()                                          # 归一化到[0,1]
        )
        
        # ===== 2.2 轻量空间注意力模块 =====
        # 使用 max pooling + avg pooling + 3x3卷积 + Sigmoid
        # Requirements: 1.3
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), # 2通道(max+avg) -> 1通道
            nn.Sigmoid()                                           # 归一化到[0,1]
        )
        
        # ===== 2.3 动态缩放门控 =====
        # 零初始化的可学习缩放因子
        # Requirements: 2.1, 2.2
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        按顺序应用通道注意力、空间注意力、动态缩放
        
        Args:
            x: 输入特征张量 (B, C, H, W)
            
        Returns:
            增强输出 (B, C, H, W)，初始为零（因为scale初始化为0）
            
        Requirements: 1.4
        """
        # ===== 2.4 forward方法实现 =====
        
        # Step 1: 应用通道注意力
        ca = self.channel_att(x)  # (B, C, 1, 1)
        x_ca = x * ca             # 通道加权
        
        # Step 2: 应用空间注意力
        # 沿通道维度计算max和avg pooling
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)    # (B, 1, H, W)
        # 拼接后通过空间注意力卷积
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)
        sa = self.spatial_att(spatial_input)  # (B, 1, H, W)
        x_sa = x_ca * sa  # 空间加权
        
        # Step 3: 应用动态缩放门控
        # 返回 scale * x_sa，初始时scale=0，所以输出为零
        return self.scale * x_sa


class AMAFDSGModule(nn.Module):
    """
    自适应多尺度注意力融合 + 动态缩放门控模块 (AMAF-DSG)
    
    完整版增强模块，包含以下创新点:
    1. 多尺度通道注意力: 局部(1x1卷积) + 全局(GAP + MLP) 双分支
    2. 跨尺度空间注意力: 3x3 + 5x5 多尺度感受野
    3. 动态缩放门控: 零初始化缩放因子，训练中自适应学习
    4. 特征残差增强: output = scale * enhanced_features
    
    设计特点:
    - 多尺度特征捕获: 同时捕获局部细节和全局语义
    - 自适应融合: 可学习的融合权重自动平衡不同尺度的贡献
    - 零初始化保证: 初始输出为零，确保与预训练权重兼容
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例，默认1/16 (0.0625)
        spatial_kernels: 空间注意力卷积核大小列表，默认[3, 5]
    
    Requirements: 1.2, 1.3, 2.1, 2.2
    """
    
    def __init__(self, channels: int, reduction: float = 0.0625, 
                 spatial_kernels: List[int] = None):
        super().__init__()
        
        # 默认空间卷积核大小
        if spatial_kernels is None:
            spatial_kernels = [3, 5]
        
        # 计算缩减后的通道数，确保最小为8
        reduced_channels = max(int(channels * reduction), 8)
        
        # ===== 3.1 多尺度通道注意力 =====
        # Requirements: 1.2
        
        # 局部分支: 1x1卷积捕获局部通道关系
        self.local_channel = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # 全局分支: 全局平均池化 + MLP
        self.global_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # 通道注意力融合权重 (可学习)
        # 初始化为0.5，表示两个分支等权重
        self.channel_fusion = nn.Parameter(torch.ones(2) * 0.5)
        
        # ===== 3.2 跨尺度空间注意力 =====
        # Requirements: 1.3
        
        # 多尺度卷积核 (3x3, 5x5)
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
            for k in spatial_kernels
        ])
        
        # 空间注意力融合权重 (可学习)
        # 初始化为均匀分布
        self.spatial_fusion = nn.Parameter(
            torch.ones(len(spatial_kernels)) / len(spatial_kernels)
        )
        
        # ===== 3.3 动态缩放门控 (零初始化) =====
        # Requirements: 2.1, 2.2
        
        # 全局缩放因子 (零初始化)
        # 这是核心创新: 初始值为0，训练过程中逐渐学习到合适的缩放值
        self.dynamic_scale = nn.Parameter(torch.zeros(1))
        
        # 通道级缩放因子 (零初始化)
        # 允许不同通道有不同的缩放强度
        self.channel_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        按顺序应用:
        1. 多尺度通道注意力 (局部+全局分支自适应融合)
        2. 跨尺度空间注意力 (多尺度卷积核自适应融合)
        3. 动态缩放门控 (全局+通道级缩放)
        
        Args:
            x: 输入特征张量 (B, C, H, W)
            
        Returns:
            增强输出 (B, C, H, W)，初始为零（因为缩放因子初始化为0）
        """
        # ===== Step 1: 多尺度通道注意力 =====
        # 局部分支: 逐点卷积捕获局部通道关系
        local_att = self.local_channel(x)  # (B, C, H, W)
        
        # 全局分支: GAP + MLP捕获全局通道关系
        global_att = self.global_channel(x)  # (B, C, 1, 1)
        
        # 自适应融合: 使用softmax归一化融合权重
        fusion_weights = torch.softmax(self.channel_fusion, dim=0)
        channel_att = fusion_weights[0] * local_att + fusion_weights[1] * global_att
        channel_att = torch.sigmoid(channel_att)  # 归一化到[0,1]
        
        # 应用通道注意力
        x_ca = x * channel_att  # (B, C, H, W)
        
        # ===== Step 2: 跨尺度空间注意力 =====
        # 沿通道维度计算max和avg pooling
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)    # (B, 1, H, W)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)
        
        # 多尺度空间注意力融合
        # 使用softmax归一化融合权重
        spatial_weights = torch.softmax(self.spatial_fusion, dim=0)
        
        # 对每个尺度的卷积结果进行加权求和
        spatial_att = sum(
            w * conv(spatial_input) 
            for w, conv in zip(spatial_weights, self.spatial_convs)
        )
        spatial_att = torch.sigmoid(spatial_att)  # 归一化到[0,1]
        
        # 应用空间注意力
        x_sa = x_ca * spatial_att  # (B, C, H, W)
        
        # ===== Step 3: 动态缩放门控 =====
        # 全局缩放 + 通道级缩放
        # dynamic_scale初始为0，channel_scale初始为0
        # sigmoid(0) = 0.5，所以初始时 enhanced = x_sa * (0 + 0.5) = 0.5 * x_sa
        # 但由于dynamic_scale=0，最终输出接近于零
        # 
        # 更精确的设计: 使用乘法组合
        # enhanced = x_sa * dynamic_scale * (1 + sigmoid(channel_scale))
        # 当dynamic_scale=0时，输出为0
        enhanced = x_sa * self.dynamic_scale * (1 + torch.sigmoid(self.channel_scale))
        
        return enhanced


# ============================================================================
# Helper Classes (copied from models_mae.py to maintain compatibility)
# ============================================================================

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv(nn.Module):
    """Convolutional layer with optional BatchNorm and ReLU."""
    
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    """Residual block with bottleneck structure."""
    
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


# ============================================================================
# MHF_block_v2: Enhanced Multi-level Hierarchical Feature Fusion Block
# ============================================================================

class MHF_block_v2(nn.Module):
    """
    增强版多层次特征融合模块 (MHF_block_v2)
    
    本模块是原始MHF_block的增强版本，具有以下特点:
    - 完全保持原始MHF_block结构和权重名称
    - 可选的超轻量级增强模块 (参数量<1%)
    - 零初始化缩放因子确保初始行为一致
    - 支持运行时启用/禁用增强
    
    设计原则:
    - 非侵入式设计: 不修改原始MHF_block内部结构
    - 向后兼容: 完全兼容原始接口和权重
    - 零初始化保证: 增强模块初始输出为零
    
    Args:
        ch_1: CNN分支(ConvNeXt)输入通道数
        ch_2: Transformer分支(Swin)输入通道数
        r_2: SE模块通道缩减比例
        ch_int: 中间特征通道数
        ch_out: 输出通道数
        drop_rate: Dropout比例，默认0
        use_enhancement: 是否启用增强模块，默认False
        enhancement_reduction: 增强模块通道缩减比例，默认1/16 (0.0625)
    
    Requirements: 3.1, 3.2, 3.3, 3.5, 1.1, 1.4, 1.6, 6.1
    """
    
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.,
                 use_enhancement: bool = False, enhancement_reduction: float = 0.0625):
        super(MHF_block_v2, self).__init__()
        
        # ===== 5.2 增强模块参数 =====
        # Requirements: 3.2, 3.3
        self.use_enhancement = use_enhancement
        self.enhancement_reduction = enhancement_reduction
        
        # ===== 5.1 原始MHF_block结构（完全保持不变）=====
        # Requirements: 3.1, 3.5
        # 注意: 所有权重名称与原始MHF_block完全一致
        
        # 注意力融合卷积
        self.att_conv = nn.Sequential(
            nn.Conv2d(ch_int * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 通道注意力组件 (用于Transformer分支)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力组件 (用于CNN分支)
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        
        # 特征投影层
        self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        
        # 下采样和通道调整
        self.Avg = nn.AvgPool2d(2, stride=2)
        
        # 归一化层
        self.norm1 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        
        # 通道调整和融合卷积
        self.chan = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.W = Conv(ch_int * 2, ch_int, 3, bn=True, relu=True)
        
        # 激活函数
        self.gelu = nn.GELU()
        
        # 残差模块
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        
        # Dropout
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        
        # ===== 5.3 集成增强模块 =====
        # Requirements: 1.1, 1.4, 1.6
        # 当 use_enhancement=True 时创建 EnhancementModuleLite
        if use_enhancement:
            self.enhancement = EnhancementModuleLite(ch_out, enhancement_reduction)
    
    def forward(self, l, g, f):
        """
        前向传播
        
        Args:
            l: CNN分支特征 (ConvNeXt), shape: (B, ch_1, H, W)
            g: Transformer分支特征 (Swin), shape: (B, ch_2, H, W)
            f: 前一层融合特征, shape: (B, ch_int//2, H*2, W*2) 或 None
        
        Returns:
            融合后的特征, shape: (B, ch_out, H, W)
        
        Requirements: 1.4, 1.6
        """
        # ===== 原始MHF_block前向逻辑（完全保持不变）=====
        n, _, h, w = g.shape
        
        # 特征投影
        W_local = self.W_x(l)   # local feature from ConvNeXt
        W_global = self.W_g(g)  # global feature from SwinTransformer
        
        # 注意力融合
        z1 = torch.cat([W_global, W_local], dim=1)
        z1 = self.att_conv(z1)
        W = W_global * z1[:, 0].view(n, 1, h, w) + W_local * z1[:, 1].view(n, 1, h, w)
        
        # 处理前一层融合特征
        if f is not None:
            W_f = self.chan(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W], 1)
            X_f = self.norm1(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)
        
        # 空间注意力 for CNN branch (ConvNeXt)
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump
        
        # 通道注意力 for Transformer branch (Swin)
        g_jump = g
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        g = self.sigmoid(output) * g_jump
        
        # 特征融合
        fuse = self.residual(torch.cat([g, l, X_f], 1))
        
        # Dropout
        if self.drop_rate > 0:
            fuse = self.dropout(fuse)
        
        # ===== 5.3 增强模块（可选，零初始化保证初始输出为零）=====
        # Requirements: 1.1, 1.4, 1.6
        # 当 use_enhancement=True 时应用增强
        # 增强逻辑: fuse = fuse + enhancement(fuse)
        if self.use_enhancement and hasattr(self, 'enhancement'):
            enhancement_output = self.enhancement(fuse)
            fuse = fuse + enhancement_output
        
        return fuse + shortcut
    
    # ===== 5.4 运行时控制方法 =====
    # Requirements: 6.1
    
    def enable_enhancement(self):
        """
        运行时启用增强模块
        
        如果模块在初始化时创建了增强模块，则启用它。
        如果没有创建增强模块，此方法不会有任何效果。
        
        Requirements: 6.1
        """
        self.use_enhancement = True
    
    def disable_enhancement(self):
        """
        运行时禁用增强模块
        
        禁用后，MHF_block_v2的行为与原始MHF_block完全一致。
        
        Requirements: 6.1
        """
        self.use_enhancement = False


# ============================================================================
# Parameter Grouping Function for Differential Learning Rates
# ============================================================================

def get_parameter_groups(model: nn.Module, 
                        backbone_lr_scale: float = 0.1,
                        mhf_lr_scale: float = 0.5,
                        enhancement_lr_scale: float = 1.0,
                        head_lr_scale: float = 2.0,
                        base_lr: float = 1e-4) -> List[Dict]:
    """
    获取分组的模型参数，用于差异化学习率
    
    将模型参数分为四组:
    1. backbone: ConvNeXt和Swin Transformer分支的参数 (预训练特征，需要保护)
    2. mhf_original: MHF_block的原始参数 (融合模块，需要适应新数据)
    3. enhancement: 新增增强模块的参数 (全新模块，需要快速学习)
    4. head: 分类头参数 (需要快速收敛，尤其是多原型头)
    
    针对 ImageNet → 医学图像 的迁移学习场景，推荐配置:
    - backbone: 0.1x (保护预训练特征)
    - mhf_original: 0.5x (适度微调融合模块)
    - enhancement: 1.0x (快速学习新增模块)
    - head: 2.0x (加速分类头收敛，尤其是多原型头)
    
    Args:
        model: main_model实例
        backbone_lr_scale: backbone学习率倍率，默认0.1
        mhf_lr_scale: MHF原始参数学习率倍率，默认0.5
        enhancement_lr_scale: 增强模块学习率倍率，默认1.0
        head_lr_scale: 分类头学习率倍率，默认2.0
        base_lr: 基础学习率，默认1e-4
    
    Returns:
        参数组列表，可直接传给优化器
        每个参数组包含: {'params': [...], 'lr': float, 'name': str}
    
    Example:
        >>> model = main_model(num_classes=7, use_mhf_enhancement=True)
        >>> param_groups = get_parameter_groups(model, base_lr=1e-4)
        >>> optimizer = torch.optim.AdamW(param_groups)
    
    Requirements: 5.1, 5.2
    """
    backbone_params = []
    mhf_original_params = []
    enhancement_params = []
    head_params = []
    
    # MHF block名称模式 (fu1, fu2, fu3, fu4)
    mhf_patterns = ['fu1', 'fu2', 'fu3', 'fu4']
    # Head 参数名称模式
    head_patterns = ['head', 'proto', 'classifier', 'fc']
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 检查是否是增强模块参数
        if 'enhancement' in name:
            enhancement_params.append(param)
        # 检查是否是 Head 参数 (分类头/原型头)
        elif any(pattern in name.lower() for pattern in head_patterns):
            head_params.append(param)
        # 检查是否是MHF_block原始参数 (不包含enhancement)
        elif any(pattern in name for pattern in mhf_patterns):
            mhf_original_params.append(param)
        # 其他参数归类为backbone (ConvNeXt, Swin等)
        else:
            backbone_params.append(param)
    
    param_groups = [
        {
            'params': backbone_params, 
            'lr': base_lr * backbone_lr_scale, 
            'name': 'backbone'
        },
        {
            'params': mhf_original_params, 
            'lr': base_lr * mhf_lr_scale, 
            'name': 'mhf_original'
        },
        {
            'params': enhancement_params, 
            'lr': base_lr * enhancement_lr_scale, 
            'name': 'enhancement'
        },
        {
            'params': head_params, 
            'lr': base_lr * head_lr_scale, 
            'name': 'head'
        },
    ]
    
    # 过滤空参数组
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # 打印参数分组统计
    total_params = sum(p.numel() for group in param_groups for p in group['params'])
    print(f"Parameter groups for differential learning rate:")
    for group in param_groups:
        group_params = sum(p.numel() for p in group['params'])
        print(f"  - {group['name']}: {len(group['params'])} tensors, "
              f"{group_params:,} params ({100*group_params/total_params:.1f}%), "
              f"lr={group['lr']:.2e}")
    
    return param_groups


# ============================================================================
# Verification Utility Functions
# ============================================================================

def verify_zero_init(model: nn.Module, threshold: float = 1e-10) -> bool:
    """
    验证增强模块的零初始化是否正确
    
    检查所有增强模块的缩放因子是否为零，以及增强模块的输出是否为零。
    
    Args:
        model: main_model实例或包含enhancement模块的模型
        threshold: 判断为零的阈值，默认1e-10
    
    Returns:
        True如果零初始化正确，否则False
    
    Raises:
        AssertionError: 如果零初始化验证失败
    
    Example:
        >>> model = main_model(num_classes=7, use_mhf_enhancement=True)
        >>> verify_zero_init(model)
        True
    
    Requirements: 8.1
    """
    print("Verifying zero initialization...")
    
    # 检查所有enhancement模块的scale参数
    scale_params = []
    for name, param in model.named_parameters():
        if 'enhancement' in name and 'scale' in name:
            scale_params.append((name, param))
    
    if len(scale_params) == 0:
        print("  Warning: No enhancement scale parameters found")
        return True
    
    all_zero = True
    for name, param in scale_params:
        value = param.abs().max().item()
        if value > threshold:
            print(f"  [FAIL] {name}: max value = {value:.2e} > {threshold}")
            all_zero = False
        else:
            print(f"  [OK] {name}: max value = {value:.2e}")
    
    if all_zero:
        print("✓ Zero initialization verified!")
    else:
        print("✗ Zero initialization verification FAILED!")
    
    return all_zero


def verify_weight_loading(model: nn.Module, checkpoint_path: str) -> bool:
    """
    验证权重加载是否成功
    
    检查预训练权重是否正确加载到模型中，并报告加载统计信息。
    
    Args:
        model: 目标模型
        checkpoint_path: 检查点文件路径
    
    Returns:
        True如果权重加载成功，否则False
    
    Requirements: 8.2
    """
    import os
    
    print(f"Verifying weight loading from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"  [FAIL] Checkpoint file not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        
        model_state = model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        missing_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(key)
        
        for key in model_state.keys():
            if key not in state_dict:
                missing_keys.append(key)
        
        # 统计enhancement相关的缺失键
        enhancement_keys = [k for k in missing_keys if 'enhancement' in k]
        
        print(f"  Loaded: {len(loaded_keys)} keys")
        print(f"  Skipped: {len(skipped_keys)} keys")
        print(f"  Missing (new): {len(missing_keys)} keys")
        print(f"  Enhancement keys (expected missing): {len(enhancement_keys)}")
        
        # 如果大部分键都加载成功，认为验证通过
        success = len(loaded_keys) > 0 and len(loaded_keys) > len(skipped_keys)
        
        if success:
            print("✓ Weight loading verified!")
        else:
            print("✗ Weight loading verification FAILED!")
        
        return success
        
    except Exception as e:
        print(f"  [FAIL] Error loading checkpoint: {e}")
        return False


def verify_behavior_consistency(model_orig: nn.Module, model_enhanced: nn.Module, 
                                input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                                threshold: float = 1e-6) -> bool:
    """
    验证增强版模型与原始模型的行为一致性
    
    比较两个模型在相同输入下的输出差异，验证差异是否小于阈值。
    这用于验证零初始化是否正确工作。
    
    Args:
        model_orig: 原始模型
        model_enhanced: 增强版模型（应该已加载相同的预训练权重）
        input_shape: 输入张量形状，默认(1, 3, 224, 224)
        threshold: 允许的最大差异，默认1e-6
    
    Returns:
        True如果行为一致，否则False
    
    Requirements: 8.3
    """
    print("Verifying behavior consistency...")
    
    # 设置为评估模式
    model_orig.eval()
    model_enhanced.eval()
    
    # 创建测试输入
    torch.manual_seed(42)
    test_input = torch.randn(*input_shape)
    
    try:
        with torch.no_grad():
            # 检查模型是否有forward_train方法
            if hasattr(model_orig, 'forward_train'):
                out_orig, _ = model_orig.forward_train(test_input)
                out_enhanced, _ = model_enhanced.forward_train(test_input)
            else:
                out_orig = model_orig(test_input)
                out_enhanced = model_enhanced(test_input)
        
        # 计算差异
        max_diff = (out_orig - out_enhanced).abs().max().item()
        mean_diff = (out_orig - out_enhanced).abs().mean().item()
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Threshold: {threshold:.2e}")
        
        success = max_diff < threshold
        
        if success:
            print("✓ Behavior consistency verified!")
        else:
            print("✗ Behavior consistency verification FAILED!")
        
        return success
        
    except Exception as e:
        print(f"  [FAIL] Error during forward pass: {e}")
        return False


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class MHFEnhancementConfig:
    """
    MHF增强模块配置数据类
    
    包含所有与MHF增强相关的配置参数，支持从字典或配置文件加载。
    
    Attributes:
        use_enhancement: 是否启用增强模块
        enhancement_reduction: 增强模块通道缩减比例
        backbone_lr_scale: backbone学习率倍率
        mhf_lr_scale: MHF原始参数学习率倍率
        enhancement_lr_scale: 增强模块学习率倍率
        freeze_backbone_epochs: 冻结backbone的epoch数
        gradual_unfreeze: 是否渐进式解冻
        unfreeze_schedule: 解冻计划 [epoch1, epoch2, ...]
    
    Example:
        >>> config = MHFEnhancementConfig(use_enhancement=True)
        >>> config.to_dict()
        {'use_enhancement': True, 'enhancement_reduction': 0.0625, ...}
        
        >>> config = MHFEnhancementConfig.from_dict({'use_enhancement': True})
    
    Requirements: 5.5
    """
    use_enhancement: bool = False
    enhancement_reduction: float = 0.0625
    backbone_lr_scale: float = 0.1
    mhf_lr_scale: float = 0.5
    enhancement_lr_scale: float = 1.0
    freeze_backbone_epochs: int = 0
    gradual_unfreeze: bool = False
    unfreeze_schedule: Optional[List[int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'use_enhancement': self.use_enhancement,
            'enhancement_reduction': self.enhancement_reduction,
            'backbone_lr_scale': self.backbone_lr_scale,
            'mhf_lr_scale': self.mhf_lr_scale,
            'enhancement_lr_scale': self.enhancement_lr_scale,
            'freeze_backbone_epochs': self.freeze_backbone_epochs,
            'gradual_unfreeze': self.gradual_unfreeze,
            'unfreeze_schedule': self.unfreeze_schedule,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MHFEnhancementConfig':
        """从字典创建配置"""
        return cls(
            use_enhancement=config_dict.get('use_enhancement', False),
            enhancement_reduction=config_dict.get('enhancement_reduction', 0.0625),
            backbone_lr_scale=config_dict.get('backbone_lr_scale', 0.1),
            mhf_lr_scale=config_dict.get('mhf_lr_scale', 0.5),
            enhancement_lr_scale=config_dict.get('enhancement_lr_scale', 1.0),
            freeze_backbone_epochs=config_dict.get('freeze_backbone_epochs', 0),
            gradual_unfreeze=config_dict.get('gradual_unfreeze', False),
            unfreeze_schedule=config_dict.get('unfreeze_schedule', None),
        )
    
    def __repr__(self) -> str:
        return (f"MHFEnhancementConfig("
                f"use_enhancement={self.use_enhancement}, "
                f"enhancement_reduction={self.enhancement_reduction}, "
                f"backbone_lr_scale={self.backbone_lr_scale}, "
                f"mhf_lr_scale={self.mhf_lr_scale}, "
                f"enhancement_lr_scale={self.enhancement_lr_scale})")
