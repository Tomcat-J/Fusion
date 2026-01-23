"""
修复后的多原型分类头损失函数
修复问题:
1. Contrastive Loss 只推开异类原型（不再推开同类原型）
2. 门控融合逻辑完整化
3. 标准化动量更新
4. 统一返回格式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class BalancedPrototypeLossFixed(nn.Module):
    """
    修复后的平衡原型损失
    
    修复内容:
    1. Contrastive Loss 只作用于异类原型
    2. 添加可选的欧氏距离支持
    """
    def __init__(
        self, 
        num_classes: int,
        max_prototypes: int,
        margin: float = 0.3,
        clst_scale: float = 0.8,
        sep_scale: float = 0.08,
        div_scale: float = 0.01,
        contrastive_scale: float = 0.1,
        use_euclidean: bool = False  # 新增：是否使用欧氏距离
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_prototypes = max_prototypes
        self.margin = margin
        self.clst_scale = clst_scale
        self.sep_scale = sep_scale
        self.div_scale = div_scale
        self.contrastive_scale = contrastive_scale
        self.use_euclidean = use_euclidean

    def _build_class_mask(
        self, 
        proto_indices: List[Tuple[int, int]], 
        device: torch.device
    ) -> torch.Tensor:
        """
        构建类别掩码矩阵
        same_class_mask[i,j] = True 表示原型 i 和 j 属于同一类
        """
        total = sum(end - start for start, end in proto_indices)
        same_class_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        
        idx = 0
        for start, end in proto_indices:
            num_protos = end - start
            same_class_mask[idx:idx+num_protos, idx:idx+num_protos] = True
            idx += num_protos
        
        return same_class_mask
    
    def compute_cluster_loss(
        self,
        distances: torch.Tensor,  # (B, num_classes, max_prototypes)
        labels: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        平衡聚类损失 - 拉近样本与同类原型
        使用 1/sqrt(n) 加权处理长尾分布
        """
        cluster_loss = 0.0
        valid_classes = 0
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            
            num_protos = proto_indices[c][1] - proto_indices[c][0]
            own_dist = distances[mask, c, :num_protos]
            min_dist = own_dist.min(dim=-1)[0]
            
            # 长尾平衡权重
            weight = 1.0 / torch.sqrt(mask.sum().float() + 1e-6)
            cluster_loss += weight * min_dist.mean()
            valid_classes += 1
        
        return cluster_loss / max(valid_classes, 1)
    
    def compute_separation_loss(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        分离损失 - 推远样本与异类原型
        使用 margin 确保足够的分离度
        """
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
        """
        多样性损失 - 鼓励同类原型捕捉不同模式
        惩罚同类原型之间过高的相似度
        """
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
            
            # 惩罚相似度 > 0.5 的情况
            div_loss += F.relu(sim[mask] - 0.5).mean()
            valid_classes += 1
        
        return div_loss / max(valid_classes, 1)
    
    def compute_contrastive_loss_fixed(
        self,
        prototypes: torch.Tensor,
        valid_mask: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        【修复】对比损失 - 只推开异类原型
        
        原问题: 推开所有原型，与 diversity loss 冲突
        修复: 只惩罚异类原型之间的相似度
        """
        valid_protos = prototypes[valid_mask]
        protos_norm = F.normalize(valid_protos, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(protos_norm, protos_norm.t())
        n = sim_matrix.size(0)
        
        # 构建同类掩码（只对有效原型）
        same_class_mask = torch.zeros(n, n, dtype=torch.bool, device=sim_matrix.device)
        idx = 0
        for start, end in proto_indices:
            num_valid = min(end - start, n - idx)  # 防止越界
            if num_valid > 0:
                same_class_mask[idx:idx+num_valid, idx:idx+num_valid] = True
                idx += num_valid
        
        # 异类掩码：非同类 且 非对角线
        diff_class_mask = ~same_class_mask & ~torch.eye(n, device=sim_matrix.device).bool()
        
        # 只惩罚异类原型的相似度
        if diff_class_mask.sum() > 0:
            contrastive_loss = sim_matrix[diff_class_mask].mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=sim_matrix.device)
        
        return contrastive_loss
    
    def forward(
        self,
        similarities: torch.Tensor,  # (B, num_classes, max_prototypes)
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]],
        valid_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算所有损失"""
        # 距离转换
        if self.use_euclidean:
            # 欧氏距离需要不同的计算方式
            distances = 2 * (1 - similarities)  # 近似欧氏距离的平方
        else:
            distances = 1 - similarities
        
        loss_dict = {}
        
        # 1. 聚类损失
        loss_dict['cluster_loss'] = self.compute_cluster_loss(
            distances, labels, proto_indices
        ) * self.clst_scale
        
        # 2. 分离损失
        loss_dict['sep_loss'] = self.compute_separation_loss(
            distances, labels, proto_indices
        ) * self.sep_scale
        
        # 3. 多样性损失
        loss_dict['div_loss'] = self.compute_diversity_loss(
            prototypes, proto_indices
        ) * self.div_scale
        
        # 4. 【修复】对比损失 - 只推开异类原型
        loss_dict['contrastive_loss'] = self.compute_contrastive_loss_fixed(
            prototypes, valid_mask, proto_indices
        ) * self.contrastive_scale
        
        loss_dict['proto_loss'] = sum(loss_dict.values())
        
        return loss_dict


class MixedAttentionBlockFixed(nn.Module):
    """
    【修复】混合注意力模块
    
    修复: 门控融合使用所有三个注意力输出
    """
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        # 轻量级自注意力（原 local_self_attn，改名更准确）
        self.lightweight_self_attn = nn.MultiheadAttention(
            emb_dim, max(1, num_heads // 2), dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 【修复】三路门控融合
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 3),
            nn.LayerNorm(emb_dim * 3),
            nn.GELU(),
            nn.Linear(emb_dim * 3, 3),  # 输出 3 个权重
        )
    
    def forward(self, proto: torch.Tensor, img_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Cross-attention
        cross_out, attn_weights = self.cross_attn(proto, img_features, img_features)
        cross_out = self.norm1(proto + self.dropout(cross_out))
        
        # 2. Self-attention
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        self_out = self.norm2(cross_out + self.dropout(self_out))
        
        # 3. Lightweight self-attention
        light_out, _ = self.lightweight_self_attn(self_out, self_out, self_out)
        light_out = self.norm3(self_out + self.dropout(light_out))
        
        # 【修复】三路门控融合
        gate_input = torch.cat([cross_out, self_out, light_out], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (B, N, 3)
        
        # 加权融合三个输出
        updated_proto = (
            gate_weights[..., 0:1] * cross_out +
            gate_weights[..., 1:2] * self_out +
            gate_weights[..., 2:3] * light_out
        )
        
        return updated_proto, attn_weights


class DynamicPrototypeManagerFixed(nn.Module):
    """
    【修复】动态原型管理器
    
    修复: 标准化 EMA 动量更新
    """
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
    def momentum_update_fixed(self, proto_idx: int, new_value: torch.Tensor):
        """
        【修复】标准 EMA 更新
        proto = momentum * proto + (1 - momentum) * new_value
        """
        new_value_norm = F.normalize(new_value, p=2, dim=-1)
        self.prototypes.data[proto_idx] = (
            self.momentum * self.prototypes.data[proto_idx] +
            (1 - self.momentum) * new_value_norm
        )
        # 重新归一化
        self.prototypes.data[proto_idx] = F.normalize(
            self.prototypes.data[proto_idx], p=2, dim=-1
        )


class HeadFixed(nn.Module):
    """
    修复后的多原型分类头
    
    修复内容:
    1. Contrastive Loss 只推开异类原型
    2. 门控融合使用所有三个注意力输出
    3. 标准化 EMA 动量更新
    4. 统一返回格式
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
        contrastive_scale: float = 0.1,
        use_euclidean: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.max_prototypes = max_prototypes
        
        # 特征投影
        self.proj = nn.Sequential(
            nn.Linear(img_feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout_rate)
        )
        
        # 动态原型管理器
        self.proto_manager = DynamicPrototypeManagerFixed(
            num_classes=num_classes,
            emb_dim=emb_dim,
            init_prototypes=num_prototypes,
            max_prototypes=max_prototypes,
            momentum=momentum
        )
        
        # 混合注意力
        self.mixed_attn = MixedAttentionBlockFixed(emb_dim, num_heads, dropout_rate)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout_rate)
        )
        self.ffn_norm = nn.LayerNorm(emb_dim)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 损失函数
        self.proto_loss_fn = BalancedPrototypeLossFixed(
            num_classes=num_classes,
            max_prototypes=max_prototypes,
            margin=margin,
            clst_scale=clst_scale,
            sep_scale=sep_scale,
            div_scale=div_scale,
            contrastive_scale=contrastive_scale,
            use_euclidean=use_euclidean
        )
        
        # 可学习的损失权重
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
        return similarities, class_emb, attn_weights
    
    def forward(
        self, 
        pool_img_features: torch.Tensor, 
        img_features: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        统一返回格式: (logits, loss, loss_dict)
        - 训练时: loss 是实际损失, loss_dict 包含各项损失
        - 推理时: loss = 0, loss_dict = None
        """
        similarities, class_emb, _ = self.forward_features(pool_img_features, img_features)
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
            
            loss_dict = {'ce_loss': ce_loss, **proto_losses, 'total_loss': total_loss}
            return logits, total_loss, loss_dict
        else:
            return logits, torch.tensor(0.0, device=logits.device), None


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 70)
    print("测试修复后的多原型分类头")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 配置
    num_classes = 7
    emb_dim = 384
    num_heads = 8
    img_feat_dim = 384
    batch_size = 16
    num_patches = 196
    
    # 初始化模型
    model = HeadFixed(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=img_feat_dim,
        num_prototypes=5,
        max_prototypes=10
    ).to(device)
    
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    pool_img_features = torch.randn(batch_size, num_patches, img_feat_dim).to(device)
    img_features = torch.randn(batch_size, img_feat_dim).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 训练模式
    print("\n训练模式测试:")
    model.train()
    logits, loss, loss_dict = model(pool_img_features, img_features, labels)
    print(f"  Logits: {logits.shape}")
    print(f"  Total Loss: {loss.item():.4f}")
    print("  损失分解:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"    {name}: {value.item():.4f}")
    
    # 推理模式
    print("\n推理模式测试:")
    model.eval()
    with torch.no_grad():
        logits, loss, loss_dict = model(pool_img_features, img_features)
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")
    
    # 梯度测试
    print("\n梯度流测试:")
    model.train()
    logits, loss, _ = model(pool_img_features, img_features, labels)
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  有梯度的参数数量: {grad_count}")
    
    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)
    
    print("""
修复总结:
=========

1. Contrastive Loss 修复
   - 原问题: 推开所有原型，与 Diversity Loss 冲突
   - 修复: 只推开异类原型，同类原型由 Diversity Loss 控制

2. 门控融合修复
   - 原问题: 只融合 local_out 和原始 proto
   - 修复: 三路 softmax 门控，融合 cross_out, self_out, light_out

3. 动量更新修复
   - 原问题: 混合 EMA 和平均，语义不清
   - 修复: 标准 EMA: proto = m * proto + (1-m) * new_value

4. 返回格式统一
   - 训练和推理都返回 (logits, loss, loss_dict)
   - 推理时 loss=0, loss_dict=None
""")
