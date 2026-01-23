"""
优化的多原型分类头 (Multi-Prototype Classification Head)
融合了以下思想:
- ProtoASNet: 动态原型 + 不确定性估计
- ProtoPNet: Push/Pull Loss + 原型可解释性
- 用户创新: 动态原型扩展 + 混合注意力 + 长尾平衡损失 + 动量推送

核心保持:
- Cross-Attention 交互机制
- 余弦距离分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class MixedAttentionBlock(nn.Module):
    """混合注意力模块: Cross + Self + Local Self Attention"""
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_self_attn = nn.MultiheadAttention(emb_dim, max(1, num_heads // 2), dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Sigmoid()
        )
    
    def forward(self, proto: torch.Tensor, img_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_out, attn_weights = self.cross_attn(proto, img_features, img_features)
        cross_out = self.norm1(proto + self.dropout(cross_out))
        
        self_out, _ = self.self_attn(cross_out, cross_out, cross_out)
        self_out = self.norm2(cross_out + self.dropout(self_out))
        
        local_out, _ = self.local_self_attn(self_out, self_out, self_out)
        local_out = self.norm3(self_out + self.dropout(local_out))
        
        gate_input = torch.cat([cross_out, self_out, local_out], dim=-1)
        gate = self.gate(gate_input)
        updated_proto = gate * local_out + (1 - gate) * proto
        
        return updated_proto, attn_weights


class DynamicPrototypeManager(nn.Module):
    """动态原型管理器: 支持动态数量、方差扩展、动量推送"""
    def __init__(self, num_classes: int, emb_dim: int, init_prototypes: int = 5,
                 max_prototypes: int = 10, var_threshold: float = 0.5, momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.max_prototypes = max_prototypes
        self.var_threshold = var_threshold
        self.momentum = momentum
        
        self.register_buffer('protos_per_class', torch.tensor([init_prototypes] * num_classes))
        
        total_max = num_classes * max_prototypes
        self.prototypes = nn.Parameter(torch.randn(total_max, emb_dim) * 0.01)
        nn.init.xavier_uniform_(self.prototypes)
        
        self.register_buffer('valid_mask', self._create_valid_mask())
        self.register_buffer('momentum_cache', torch.zeros(total_max, emb_dim))
        self.register_buffer('cache_count', torch.zeros(total_max))
    
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
    def expand_prototypes(self, class_idx: int, new_proto: torch.Tensor):
        if self.protos_per_class[class_idx] >= self.max_prototypes:
            return False
        new_idx = class_idx * self.max_prototypes + self.protos_per_class[class_idx].item()
        self.prototypes.data[new_idx] = F.normalize(new_proto, p=2, dim=-1)
        self.protos_per_class[class_idx] += 1
        self.valid_mask[new_idx] = True
        return True
    
    @torch.no_grad()
    def momentum_update(self, proto_idx: int, new_value: torch.Tensor):
        self.momentum_cache[proto_idx] = self.momentum * self.momentum_cache[proto_idx] + (1 - self.momentum) * new_value
        self.cache_count[proto_idx] += 1
    
    @torch.no_grad()
    def apply_momentum_updates(self):
        for idx in range(self.prototypes.size(0)):
            if self.cache_count[idx] > 0 and self.valid_mask[idx]:
                self.prototypes.data[idx] = F.normalize(self.momentum_cache[idx] / self.cache_count[idx], p=2, dim=-1)
        self.momentum_cache.zero_()
        self.cache_count.zero_()


class BalancedPrototypeLoss(nn.Module):
    """平衡原型损失: 处理长尾分布"""
    def __init__(self, num_classes: int, margin: float = 0.3, clst_scale: float = 0.8,
                 sep_scale: float = 0.08, div_scale: float = 0.01, contrastive_scale: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.clst_scale = clst_scale
        self.sep_scale = sep_scale
        self.div_scale = div_scale
        self.contrastive_scale = contrastive_scale
    
    def forward(self, similarities: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor,
                proto_indices: List[Tuple[int, int]], valid_mask: torch.Tensor, max_prototypes: int) -> Dict[str, torch.Tensor]:
        distances = 1 - similarities
        loss_dict = {}
        
        # Balanced cluster loss
        cluster_loss = 0.0
        valid_classes = 0
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            num_protos = proto_indices[c][1] - proto_indices[c][0]
            own_dist = distances[mask, c, :num_protos]
            min_dist = own_dist.min(dim=-1)[0]
            weight = 1.0 / torch.sqrt(mask.sum().float() + 1e-6)
            cluster_loss += weight * min_dist.mean()
            valid_classes += 1
        loss_dict['cluster_loss'] = (cluster_loss / max(valid_classes, 1)) * self.clst_scale
        
        # Separation loss
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
        loss_dict['sep_loss'] = (sep_loss / max(valid_classes, 1)) * self.sep_scale
        
        # Diversity loss
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
        loss_dict['div_loss'] = (div_loss / max(valid_classes, 1)) * self.div_scale
        
        # Contrastive loss
        valid_protos = prototypes[valid_mask]
        protos_norm = F.normalize(valid_protos, p=2, dim=-1)
        sim_matrix = torch.mm(protos_norm, protos_norm.t())
        n = sim_matrix.size(0)
        mask = ~torch.eye(n, device=sim_matrix.device).bool()
        loss_dict['contrastive_loss'] = sim_matrix[mask].mean() * self.contrastive_scale
        
        loss_dict['proto_loss'] = sum(loss_dict.values())
        return loss_dict



class Head(nn.Module):
    """
    优化的多原型分类头 - 兼容原有接口
    """
    def __init__(self, num_classes: int, emb_dim: int, num_heads: int, img_feat_dim: int,
                 num_prototypes: int = 5, max_prototypes: int = 10, dropout_rate: float = 0.1,
                 var_threshold: float = 0.5, momentum: float = 0.9, margin: float = 0.3,
                 temperature: float = 10.0, clst_scale: float = 0.8, sep_scale: float = 0.08,
                 div_scale: float = 0.01, contrastive_scale: float = 0.1):
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
        self.proto_manager = DynamicPrototypeManager(
            num_classes=num_classes, emb_dim=emb_dim, init_prototypes=num_prototypes,
            max_prototypes=max_prototypes, var_threshold=var_threshold, momentum=momentum
        )
        
        # 混合注意力
        self.mixed_attn = MixedAttentionBlock(emb_dim, num_heads, dropout_rate)
        
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
        self.proto_loss_fn = BalancedPrototypeLoss(
            num_classes=num_classes, margin=margin, clst_scale=clst_scale,
            sep_scale=sep_scale, div_scale=div_scale, contrastive_scale=contrastive_scale
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
    
    def forward(self, pool_img_features: torch.Tensor, img_features: torch.Tensor, 
                labels: Optional[torch.Tensor] = None):
        similarities, class_emb, _ = self.forward_features(pool_img_features, img_features)
        class_logits = self.aggregate_similarities(similarities)
        logits = class_logits * self.temperature
        
        if self.training and labels is not None:
            # 计算损失
            ce_loss = F.cross_entropy(logits, labels)
            proto_losses = self.proto_loss_fn(
                similarities=similarities, labels=labels,
                prototypes=self.proto_manager.prototypes,
                proto_indices=self.proto_manager.get_prototype_indices(),
                valid_mask=self.proto_manager.valid_mask,
                max_prototypes=self.max_prototypes
            )
            
            total_loss = (
                torch.exp(-self.log_sigma_ce) * ce_loss + self.log_sigma_ce +
                torch.exp(-self.log_sigma_proto) * proto_losses['proto_loss'] + self.log_sigma_proto
            )
            return logits, total_loss
        else:
            return logits, torch.tensor(0.0, device=logits.device)


if __name__ == "__main__":
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Head(
        num_classes=7, emb_dim=768, num_heads=12, img_feat_dim=768,
        num_prototypes=5, max_prototypes=10
    ).to(device)
    
    pool_img_features = torch.randn(4, 49, 768).to(device)
    img_features = torch.randn(4, 1, 768).to(device)
    labels = torch.randint(0, 7, (4,)).to(device)
    
    model.train()
    logits, loss = model(pool_img_features, img_features, labels)
    print(f"Train - Logits: {logits.shape}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(pool_img_features, img_features)
    print(f"Eval - Logits: {logits.shape}")
    print("Test passed!")
