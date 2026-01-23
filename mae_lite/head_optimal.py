"""
最优多原型分类头 (Optimal Multi-Prototype Classification Head)

基于以下分析设计:
1. ProtoPNet: Push/Pull 机制
2. 长尾数据平衡: Class-Balanced 加权
3. 修复 Contrastive Loss: 只推开异类原型

损失函数组成:
L_total = L_ce + λ_clst * L_cluster + λ_sep * L_separation 
        + λ_div * L_diversity + λ_cont * L_contrastive_fixed

注意: 已移除 Abstention Loss，简化实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class OptimalPrototypeLoss(nn.Module):
    """
    最优原型损失函数
    
    修复内容:
    1. Contrastive Loss 只作用于异类原型
    2. 支持 Class-Balanced 加权（适合严重长尾）
    """
    def __init__(
        self,
        num_classes: int,
        max_prototypes: int,
        margin: float = 0.3,
        clst_scale: float = 0.8,
        sep_scale: float = 0.08,
        div_scale: float = 0.01,
        contrastive_scale: float = 0.05,
        use_class_balanced: bool = False,
        cb_beta: float = 0.9999,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_prototypes = max_prototypes
        self.margin = margin
        self.clst_scale = clst_scale
        self.sep_scale = sep_scale
        self.div_scale = div_scale
        self.contrastive_scale = contrastive_scale
        self.use_class_balanced = use_class_balanced
        self.cb_beta = cb_beta

    def _get_class_weight(self, class_count: torch.Tensor) -> torch.Tensor:
        """计算类别权重"""
        if self.use_class_balanced:
            effective_num = 1.0 - torch.pow(self.cb_beta, class_count.float())
            weights = (1.0 - self.cb_beta) / (effective_num + 1e-6)
        else:
            weights = 1.0 / torch.sqrt(class_count.float() + 1e-6)
        return weights / weights.sum() * self.num_classes
    
    def compute_cluster_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """平衡聚类损失 - 拉近样本与同类原型"""
        cluster_loss = 0.0
        valid_classes = 0
        class_counts = []
        
        for c in range(self.num_classes):
            mask = (labels == c)
            class_counts.append(mask.sum())
            if mask.sum() == 0:
                continue
            num_protos = proto_indices[c][1] - proto_indices[c][0]
            own_dist = distances[mask, c, :num_protos]
            min_dist = own_dist.min(dim=-1)[0]
            cluster_loss += min_dist.mean()
            valid_classes += 1
        
        if valid_classes > 0:
            class_counts = torch.stack(class_counts)
            weights = self._get_class_weight(class_counts)
            
            weighted_loss = 0.0
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                num_protos = proto_indices[c][1] - proto_indices[c][0]
                own_dist = distances[mask, c, :num_protos]
                min_dist = own_dist.min(dim=-1)[0]
                weighted_loss += weights[c] * min_dist.mean()
            return weighted_loss / valid_classes
        return torch.tensor(0.0, device=distances.device)

    def compute_separation_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """分离损失 - 推远样本与异类原型"""
        sep_loss = 0.0
        valid_classes = 0
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            other_dists = []
            for other_c in range(self.num_classes):
                if other_c == c:
                    continue
                num_protos = proto_indices[other_c][1] - proto_indices[other_c][0]
                if num_protos > 0:
                    other_dists.append(distances[mask, other_c, :num_protos])
            if other_dists:
                all_other_dist = torch.cat(other_dists, dim=-1)
                nearest_other = all_other_dist.min(dim=-1)[0]
                sep_loss += F.relu(self.margin - nearest_other).mean()
                valid_classes += 1
        return sep_loss / max(valid_classes, 1)
    
    def compute_diversity_loss(
        self,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """多样性损失 - 鼓励同类原型捕捉不同模式"""
        protos_norm = F.normalize(prototypes, p=2, dim=-1)
        div_loss = 0.0
        valid_classes = 0
        
        for start, end in proto_indices:
            num_protos = end - start
            if num_protos <= 1:
                continue
            class_protos = protos_norm[start:end]
            sim = torch.mm(class_protos, class_protos.t())
            mask = ~torch.eye(num_protos, device=sim.device).bool()
            div_loss += F.relu(sim[mask] - 0.5).mean()
            valid_classes += 1
        return div_loss / max(valid_classes, 1)

    def compute_contrastive_loss(
        self,
        prototypes: torch.Tensor,
        valid_mask: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        【核心修复】对比损失 - 只推开异类原型
        原问题: 推开所有原型，与 diversity loss 冲突
        修复: 只惩罚异类原型之间的相似度
        """
        valid_protos = prototypes[valid_mask]
        n = valid_protos.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=prototypes.device)
        
        protos_norm = F.normalize(valid_protos, p=2, dim=-1)
        sim_matrix = torch.mm(protos_norm, protos_norm.t())
        
        # 构建同类掩码
        same_class_mask = torch.zeros(n, n, dtype=torch.bool, device=prototypes.device)
        idx = 0
        for start, end in proto_indices:
            num_valid = end - start
            if idx + num_valid > n:
                num_valid = n - idx
            if num_valid > 0:
                same_class_mask[idx:idx+num_valid, idx:idx+num_valid] = True
                idx += num_valid
            if idx >= n:
                break
        
        eye = torch.eye(n, device=prototypes.device).bool()
        diff_class_mask = ~same_class_mask & ~eye
        
        if diff_class_mask.sum() > 0:
            return sim_matrix[diff_class_mask].mean()
        return torch.tensor(0.0, device=prototypes.device)
    
    def forward(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]],
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算所有损失"""
        distances = 1 - similarities
        loss_dict = {}
        
        loss_dict['cluster_loss'] = self.compute_cluster_loss(
            distances, labels, proto_indices) * self.clst_scale
        loss_dict['sep_loss'] = self.compute_separation_loss(
            distances, labels, proto_indices) * self.sep_scale
        loss_dict['div_loss'] = self.compute_diversity_loss(
            prototypes, proto_indices) * self.div_scale
        loss_dict['contrastive_loss'] = self.compute_contrastive_loss(
            prototypes, valid_mask, proto_indices) * self.contrastive_scale
        loss_dict['proto_loss'] = sum(loss_dict.values())
        return loss_dict


class MixedAttentionBlockOptimal(nn.Module):
    """优化的混合注意力模块 - 三路 softmax 门控融合"""
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.light_attn = nn.MultiheadAttention(emb_dim, max(1, num_heads // 2), dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 3),
        )
    
    def forward(self, proto: torch.Tensor, img_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_out, attn_weights = self.cross_attn(proto, img_features, img_features)
        cross_out = self.norm1(proto + self.dropout(cross_out))
        
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        self_out = self.norm2(cross_out + self.dropout(self_out))
        
        light_out, _ = self.light_attn(self_out, self_out, self_out)
        light_out = self.norm3(self_out + self.dropout(light_out))
        
        gate_input = torch.cat([cross_out, self_out, light_out], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        
        updated_proto = (
            gate_weights[..., 0:1] * cross_out +
            gate_weights[..., 1:2] * self_out +
            gate_weights[..., 2:3] * light_out
        )
        return updated_proto, attn_weights


class DynamicPrototypeManagerOptimal(nn.Module):
    """优化的动态原型管理器 - 标准 EMA 动量更新"""
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        init_prototypes: int = 5,
        max_prototypes: int = 10,
        momentum: float = 0.9
    ):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.max_prototypes = max_prototypes
        self.momentum = momentum
        
        self.register_buffer('protos_per_class', torch.tensor([init_prototypes] * num_classes))
        
        total_max = num_classes * max_prototypes
        self.prototypes = nn.Parameter(torch.randn(total_max, emb_dim) * 0.01)
        nn.init.xavier_uniform_(self.prototypes)
        
        self.register_buffer('valid_mask', self._create_valid_mask())
    
    def _create_valid_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_classes * self.max_prototypes, dtype=torch.bool)
        for c in range(self.num_classes):
            start = c * self.max_prototypes
            end = start + self.protos_per_class[c].item()
            mask[start:end] = True
        return mask
    
    def get_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)
    
    def get_prototype_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for c in range(self.num_classes):
            start = c * self.max_prototypes
            end = start + self.protos_per_class[c].item()
            indices.append((start, end))
        return indices
    
    @torch.no_grad()
    def expand_prototype(self, class_idx: int, new_proto: torch.Tensor) -> bool:
        if self.protos_per_class[class_idx] >= self.max_prototypes:
            return False
        new_idx = class_idx * self.max_prototypes + self.protos_per_class[class_idx].item()
        self.prototypes.data[new_idx] = F.normalize(new_proto, p=2, dim=-1)
        self.protos_per_class[class_idx] += 1
        self.valid_mask[new_idx] = True
        return True
    
    @torch.no_grad()
    def momentum_update(self, proto_idx: int, new_value: torch.Tensor):
        """标准 EMA 更新"""
        new_value_norm = F.normalize(new_value, p=2, dim=-1)
        self.prototypes.data[proto_idx] = (
            self.momentum * self.prototypes.data[proto_idx] +
            (1 - self.momentum) * new_value_norm
        )
        self.prototypes.data[proto_idx] = F.normalize(
            self.prototypes.data[proto_idx], p=2, dim=-1
        )


class HeadOptimal(nn.Module):
    """
    最优多原型分类头
    
    核心改进:
    1. Contrastive Loss 只推开异类原型
    2. 三路门控融合
    3. 标准 EMA 动量更新
    4. 可选 Class-Balanced 加权
    
    返回格式:
    - 训练: (logits, total_loss, loss_dict)
    - 推理: (logits, 0.0, {})
    """
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        num_heads: int,
        img_feat_dim: int,
        num_prototypes: int = 5,
        max_prototypes: int = 10,
        dropout_rate: float = 0.1,
        momentum: float = 0.9,
        margin: float = 0.3,
        temperature: float = 10.0,
        clst_scale: float = 0.8,
        sep_scale: float = 0.08,
        div_scale: float = 0.01,
        contrastive_scale: float = 0.05,
        use_class_balanced: bool = False,
        cb_beta: float = 0.9999,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.max_prototypes = max_prototypes
        
        self.proj = nn.Sequential(
            nn.Linear(img_feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.proto_manager = DynamicPrototypeManagerOptimal(
            num_classes=num_classes,
            emb_dim=emb_dim,
            init_prototypes=num_prototypes,
            max_prototypes=max_prototypes,
            momentum=momentum
        )
        
        self.mixed_attn = MixedAttentionBlockOptimal(emb_dim, num_heads, dropout_rate)
        
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout_rate)
        )
        self.ffn_norm = nn.LayerNorm(emb_dim)
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.proto_loss_fn = OptimalPrototypeLoss(
            num_classes=num_classes,
            max_prototypes=max_prototypes,
            margin=margin,
            clst_scale=clst_scale,
            sep_scale=sep_scale,
            div_scale=div_scale,
            contrastive_scale=contrastive_scale,
            use_class_balanced=use_class_balanced,
            cb_beta=cb_beta,
        )
        
        self.log_sigma_ce = nn.Parameter(torch.zeros(1))
        self.log_sigma_proto = nn.Parameter(torch.zeros(1))

    def compute_similarities(self, img_features: torch.Tensor, class_emb: torch.Tensor) -> torch.Tensor:
        B = img_features.size(0)
        img_norm = F.normalize(img_features, p=2, dim=-1)
        class_norm = F.normalize(class_emb, p=2, dim=-1)
        sim = torch.bmm(class_norm, img_norm.unsqueeze(-1)).squeeze(-1)
        return sim.view(B, self.num_classes, self.max_prototypes)
    
    def aggregate_similarities(self, similarities: torch.Tensor) -> torch.Tensor:
        proto_indices = self.proto_manager.get_prototype_indices()
        B = similarities.size(0)
        logits = torch.zeros(B, self.num_classes, device=similarities.device)
        for c in range(self.num_classes):
            start, end = proto_indices[c]
            num_protos = end - start
            class_sim = similarities[:, c, :num_protos]
            logits[:, c] = class_sim.max(dim=-1)[0]
        return logits
    
    def forward_features(self, pool_img_features: torch.Tensor, img_features: torch.Tensor):
        B, N, C = pool_img_features.shape
        
        if img_features.dim() == 3:
            img_features = img_features.mean(dim=1)
        
        img_features = self.proj(img_features)
        pool_img_features = self.proj(pool_img_features)
        pool_flat = pool_img_features.reshape(1, B * N, self.emb_dim)
        
        prototypes = self.proto_manager.get_prototypes()
        proto_emb = prototypes.unsqueeze(0)
        
        class_emb, attn_weights = self.mixed_attn(proto_emb, pool_flat)
        ffn_out = self.ffn(class_emb)
        class_emb = self.ffn_norm(class_emb + ffn_out)
        class_emb = class_emb.expand(B, -1, -1)
        
        similarities = self.compute_similarities(img_features, class_emb)
        return similarities, class_emb, attn_weights, img_features
    
    def forward(
        self,
        pool_img_features: torch.Tensor,
        img_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        返回格式: (logits, loss, loss_dict)
        """
        similarities, class_emb, _, proj_img_features = self.forward_features(
            pool_img_features, img_features
        )
        class_logits = self.aggregate_similarities(similarities)
        logits = class_logits * self.temperature
        
        if self.training and labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            
            proto_losses = self.proto_loss_fn(
                similarities=similarities,
                labels=labels,
                prototypes=self.proto_manager.prototypes,
                proto_indices=self.proto_manager.get_prototype_indices(),
                valid_mask=self.proto_manager.valid_mask
            )
            
            total_loss = (
                torch.exp(-self.log_sigma_ce) * ce_loss + self.log_sigma_ce +
                torch.exp(-self.log_sigma_proto) * proto_losses['proto_loss'] + self.log_sigma_proto
            )
            
            loss_dict = {
                'ce_loss': ce_loss,
                **proto_losses,
                'total_loss': total_loss
            }
            return logits, total_loss, loss_dict
        else:
            return logits, torch.tensor(0.0, device=logits.device), {}


# ============ 兼容接口 ============
class Head(HeadOptimal):
    """
    兼容原有接口的简化版本
    返回格式: (logits, loss) - 与原 head.py 一致
    """
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        num_heads: int,
        img_feat_dim: int,
        num_prototypes: int = 5,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            emb_dim=emb_dim,
            num_heads=num_heads,
            img_feat_dim=img_feat_dim,
            num_prototypes=num_prototypes,
            max_prototypes=num_prototypes * 2,
            dropout_rate=dropout_rate,
            **kwargs
        )
    
    def forward(self, pool_img_features, img_features, labels=None):
        """兼容原有接口: 返回 (logits, loss)"""
        if self.training:
            logits, loss, _ = super().forward(pool_img_features, img_features, labels)
            return logits, loss
        else:
            logits, _, _ = super().forward(pool_img_features, img_features)
            return logits, torch.tensor(0.0, device=logits.device)


# ============ 工厂函数 ============
def create_optimal_head(
    num_classes: int,
    emb_dim: int,
    num_heads: int = 8,
    num_prototypes: int = 5,
    use_class_balanced: bool = False,
    **kwargs
) -> HeadOptimal:
    """
    通用工厂函数 - 适用于任何模型
    
    Args:
        num_classes: 类别数
        emb_dim: 特征维度 (模型输出的特征维度)
        num_heads: 注意力头数
        num_prototypes: 每类初始原型数
        use_class_balanced: 是否使用 Class-Balanced 加权
        **kwargs: 其他参数传递给 HeadOptimal
    
    Returns:
        HeadOptimal 实例
    """
    return HeadOptimal(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=emb_dim,
        num_prototypes=num_prototypes,
        max_prototypes=num_prototypes * 2,
        use_class_balanced=use_class_balanced,
        **kwargs
    )


def create_optimal_head_for_main_model(
    num_classes: int,
    conv_dims: List[int] = None,
    emb_dim: int = 768,
    num_heads: int = 12,
    num_prototypes: int = 5,
    use_class_balanced: bool = False,
    **kwargs
) -> HeadOptimal:
    """
    为 main_model (CNN+Transformer 融合模型) 创建最优分类头
    
    Args:
        num_classes: 类别数
        conv_dims: ConvNeXt 各阶段通道数，如 [96, 192, 384, 768]
        emb_dim: 特征维度，如果提供 conv_dims 则使用 conv_dims[-1]
        num_heads: 注意力头数
        num_prototypes: 每类初始原型数
        use_class_balanced: 是否使用 Class-Balanced 加权
        **kwargs: 其他参数传递给 HeadOptimal
    
    Returns:
        HeadOptimal 实例
    """
    if conv_dims is not None:
        emb_dim = conv_dims[-1]
    
    return HeadOptimal(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=emb_dim,
        num_prototypes=num_prototypes,
        max_prototypes=num_prototypes * 2,
        use_class_balanced=use_class_balanced,
        **kwargs
    )


# ============ 辅助函数: 检测模型是否使用多原型头 ============
def has_multi_prototype_head(model: nn.Module) -> bool:
    """
    检测模型是否使用多原型分类头
    
    用于 timm_imagenet_exp.py 中判断使用哪种损失计算方式:
    - 多原型头: 使用 head 内置的损失
    - 标准线性头: 使用 train_loss_fn (CE/CB_loss)
    
    Args:
        model: 模型实例
    
    Returns:
        True 如果模型使用多原型头
    """
    # 检查是否有 head 属性
    if hasattr(model, 'head'):
        head = model.head
        # 检查是否是 HeadOptimal 或其子类
        if isinstance(head, (HeadOptimal, Head)):
            return True
        # 检查是否有 proto_manager 属性 (原 head.py 的特征)
        if hasattr(head, 'proto_manager'):
            return True
    return False


def get_head_loss_from_output(output, default_loss_fn=None, logits=None, target=None):
    """
    从模型输出中提取损失
    
    处理不同模型的返回格式:
    1. 多原型头模型: forward 返回 (loss, extra_dict)
    2. 标准模型: forward 返回 logits，需要用 default_loss_fn 计算损失
    
    Args:
        output: 模型 forward 的输出
        default_loss_fn: 默认损失函数 (用于标准模型)
        logits: 如果 output 是 logits，直接使用
        target: 标签
    
    Returns:
        (loss, extra_dict)
    """
    # 如果 output 是 tuple 且第一个元素是标量 tensor (loss)
    if isinstance(output, tuple) and len(output) >= 2:
        first = output[0]
        if isinstance(first, torch.Tensor) and first.dim() == 0:
            # 返回格式是 (loss, extra_dict)
            return first, output[1] if len(output) > 1 else {}
        elif isinstance(first, torch.Tensor) and first.dim() >= 1:
            # 返回格式是 (logits, head_loss) 或 (logits, head_loss, loss_dict)
            logits = first
            head_loss = output[1] if len(output) > 1 else torch.tensor(0.0)
            if default_loss_fn is not None and target is not None:
                ce_loss = default_loss_fn(logits, target)
                total_loss = ce_loss + head_loss
                return total_loss, {'ce_loss': ce_loss, 'head_loss': head_loss}
            return head_loss, {}
    
    # 如果 output 是单个 tensor (logits)
    if isinstance(output, torch.Tensor):
        if default_loss_fn is not None and target is not None:
            loss = default_loss_fn(output, target)
            return loss, {}
    
    return torch.tensor(0.0), {}


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 70)
    print("测试最优多原型分类头 (HeadOptimal) - 无 Abstention Loss 版本")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_classes = 7
    emb_dim = 384
    num_heads = 8
    img_feat_dim = 384
    batch_size = 16
    num_patches = 196
    
    print(f"\n配置: num_classes={num_classes}, emb_dim={emb_dim}, device={device}")
    
    # 测试1: 基础版本
    print("\n" + "-" * 50)
    print("测试1: 基础版本")
    model = HeadOptimal(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=img_feat_dim,
        num_prototypes=5,
        max_prototypes=10,
        use_class_balanced=False,
    ).to(device)
    
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    pool_img_features = torch.randn(batch_size, num_patches, img_feat_dim).to(device)
    img_features = torch.randn(batch_size, img_feat_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    model.train()
    logits, loss, loss_dict = model(pool_img_features, img_features, labels)
    print(f"  Logits: {logits.shape}, Total Loss: {loss.item():.4f}")
    print("  损失分解:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"    {name}: {value.item():.4f}")
    
    # 测试2: Class-Balanced 加权
    print("\n" + "-" * 50)
    print("测试2: Class-Balanced 加权 (严重长尾)")
    model_cb = HeadOptimal(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=img_feat_dim,
        num_prototypes=5,
        max_prototypes=10,
        use_class_balanced=True,
        cb_beta=0.9999,
    ).to(device)
    
    model_cb.train()
    logits, loss, _ = model_cb(pool_img_features, img_features, labels)
    print(f"  Total Loss: {loss.item():.4f}")
    
    # 测试3: 推理模式
    print("\n" + "-" * 50)
    print("测试3: 推理模式")
    model.eval()
    with torch.no_grad():
        logits, _, info = model(pool_img_features, img_features)
    print(f"  Logits: {logits.shape}")
    
    # 测试4: 兼容接口
    print("\n" + "-" * 50)
    print("测试4: 兼容接口 (Head)")
    simple_head = Head(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=img_feat_dim,
        num_prototypes=5,
    ).to(device)
    
    simple_head.train()
    logits, loss = simple_head(pool_img_features, img_features, labels)
    print(f"  Logits: {logits.shape}, Loss: {loss.item():.4f}")
    
    # 测试5: main_model 兼容性 (768 维特征)
    print("\n" + "-" * 50)
    print("测试5: main_model 兼容性 (768 维特征)")
    main_model_head = create_optimal_head_for_main_model(
        num_classes=7,
        conv_dims=[96, 192, 384, 768],
        num_heads=12,
        num_prototypes=5,
        use_class_balanced=True,
    ).to(device)
    
    x_fu = torch.randn(8, 1, 768).to(device)
    x_fu1 = torch.randn(8, 49, 768).to(device)
    labels_7 = torch.randint(0, 7, (8,)).to(device)
    
    main_model_head.train()
    logits, loss, _ = main_model_head(x_fu1, x_fu, labels_7)
    print(f"  输入 pool_features: {x_fu1.shape}")
    print(f"  输入 img_features: {x_fu.shape}")
    print(f"  输出 Logits: {logits.shape}, Total Loss: {loss.item():.4f}")
    print("  ✓ 与 main_model 完全兼容!")
    
    # 测试6: 检测函数
    print("\n" + "-" * 50)
    print("测试6: has_multi_prototype_head 检测函数")
    
    class DummyModel(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
    
    model_with_proto_head = DummyModel(simple_head)
    model_with_linear_head = DummyModel(nn.Linear(768, 7))
    
    print(f"  多原型头模型: {has_multi_prototype_head(model_with_proto_head)}")
    print(f"  线性头模型: {has_multi_prototype_head(model_with_linear_head)}")
    
    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
