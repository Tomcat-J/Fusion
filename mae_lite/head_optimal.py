"""
最优多原型分类头 (Optimal Multi-Prototype Classification Head)

基于以下分析设计:
1. ProtoPNet: Push/Pull 机制
2. 长尾数据平衡: Class-Balanced 加权
3. 软正交损失: 替代 Contrastive + Diversity，减少损失冲突
4. 局部一致性损失: 同类样本原型激活分布一致性约束

损失函数组成 (优化后):
L_total = L_ce + λ_clst * L_cluster + λ_sep * L_separation + λ_orth * L_soft_orth + λ_lc * L_local_consistency

变更记录:
- 移除 Contrastive Loss (与 Separation 功能重叠)
- 移除 Diversity Loss (被软正交替代)
- 新增 Soft Orthogonality Loss (类内原型软正交约束)
- 新增 Local Consistency Loss (同类样本原型激活一致性)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class OptimalPrototypeLoss(nn.Module):
    """
    最优原型损失函数 (优化版)
    
    变更:
    1. 移除 Contrastive Loss (与 Separation 功能重叠)
    2. 用软正交损失替代 Diversity Loss
    3. 支持 Class-Balanced 加权（适合严重长尾）
    4. CB 加权同时作用于 CE Loss 和 Cluster Loss
    """
    def __init__(
        self,
        num_classes: int,
        max_prototypes: int,
        margin: float = 0.3,
        clst_scale: float = 0.8,
        sep_scale: float = 0.12,      # 提高 (原 0.08)
        orth_scale: float = 0.08,     # 新增: 软正交权重
        orth_threshold: float = 0.3,  # 新增: 软正交阈值
        local_consistency_scale: float = 0.05,  # 新增: 局部一致性权重
        local_consistency_temp: float = 0.1,    # 新增: 局部一致性温度
        use_class_balanced: bool = False,
        cb_beta: float = 0.9999,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_prototypes = max_prototypes
        self.margin = margin
        self.clst_scale = clst_scale
        self.sep_scale = sep_scale
        self.orth_scale = orth_scale
        self.orth_threshold = orth_threshold
        self.local_consistency_scale = local_consistency_scale
        self.local_consistency_temp = local_consistency_temp
        self.use_class_balanced = use_class_balanced
        self.cb_beta = cb_beta
        
        # 缓存 CB 权重 (在第一次 forward 时计算)
        self._cb_weights: Optional[torch.Tensor] = None

    def _get_class_weight(self, class_count: torch.Tensor) -> torch.Tensor:
        """计算类别权重"""
        if self.use_class_balanced:
            effective_num = 1.0 - torch.pow(self.cb_beta, class_count.float())
            weights = (1.0 - self.cb_beta) / (effective_num + 1e-6)
        else:
            weights = 1.0 / torch.sqrt(class_count.float() + 1e-6)
        return weights / weights.sum() * self.num_classes
    
    def get_ce_weights(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        获取 CE Loss 的类别权重
        【新增】用于在 HeadOptimal.forward 中对 CE Loss 加权
        """
        if not self.use_class_balanced:
            return None
        
        # 统计当前 batch 的类别分布
        class_counts = torch.zeros(self.num_classes, device=labels.device)
        for c in range(self.num_classes):
            class_counts[c] = (labels == c).sum().float() + 1e-6  # 避免除零
        
        return self._get_class_weight(class_counts)
    
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
    
    def compute_soft_orthogonality_loss(
        self,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        软正交损失 - 替代 Contrastive + Diversity
        
        目标: 类内原型之间保持低相似度，但不强制严格正交
        优点:
        1. 比严格正交更灵活 (允许一定相似度)
        2. 统一替代 div_loss + contrastive_loss
        3. 只约束类内，不干扰异类分离
        
        Args:
            prototypes: 所有原型 (num_classes * max_prototypes, emb_dim)
            proto_indices: 每类原型的索引范围
        
        Returns:
            软正交损失值
        """
        protos_norm = F.normalize(prototypes, p=2, dim=-1)
        orth_loss = 0.0
        valid_classes = 0
        
        for start, end in proto_indices:
            num_protos = end - start
            if num_protos <= 1:
                continue
            
            class_protos = protos_norm[start:end]
            gram = torch.mm(class_protos, class_protos.t())
            
            # 只看非对角线元素 (原型之间的相似度)
            mask = ~torch.eye(num_protos, device=gram.device).bool()
            off_diag = gram[mask]
            
            # 软约束: 只惩罚 |sim| > threshold
            # 允许原型之间有一定相似度，但不能太高
            orth_loss += F.relu(off_diag.abs() - self.orth_threshold).pow(2).mean()
            valid_classes += 1
        
        return orth_loss / max(valid_classes, 1)
    
    def compute_local_consistency_loss(
        self,
        local_feats: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]],
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        局部一致性损失 - 同类样本的原型激活分布应趋于一致
        
        理论依据:
        - 同类样本的"最相似 patch"应聚集到相似的原型
        - 强化多原型头的局部对齐特性
        - 与软正交配合，提升类内一致性
        
        实现方式 (可微分):
        1. 对每个样本的局部 patches，计算与同类原型的相似度
        2. 用 softmax 得到软分配分布 (可微分)
        3. 同类样本的分布应相似 → 最小化 KL 散度
        
        Args:
            local_feats: 局部 patch 特征 (B, L, emb_dim)，L=49 for 7x7 patches
            labels: 标签 (B,)
            prototypes: 所有原型 (num_classes * max_prototypes, emb_dim)
            proto_indices: 每类原型的索引范围
            temperature: softmax 温度 (越小分布越尖锐)
        
        Returns:
            局部一致性损失值
        """
        B, L, D = local_feats.shape
        device = local_feats.device
        
        # 归一化
        local_feats_norm = F.normalize(local_feats, p=2, dim=-1)  # (B, L, D)
        protos_norm = F.normalize(prototypes, p=2, dim=-1)
        
        consistency_loss = 0.0
        valid_classes = 0
        
        for c in range(self.num_classes):
            mask = (labels == c)
            num_samples = mask.sum().item()
            
            if num_samples < 2:
                # 需要至少 2 个样本才能计算一致性
                continue
            
            # 获取该类的原型
            start, end = proto_indices[c]
            num_protos = end - start
            if num_protos == 0:
                continue
            
            class_protos = protos_norm[start:end]  # (num_protos, D)
            
            # 获取该类样本的局部特征
            class_local_feats = local_feats_norm[mask]  # (N_c, L, D)
            N_c = class_local_feats.size(0)
            
            # 计算每个 patch 与每个原型的相似度
            # class_local_feats: (N_c, L, D)
            # class_protos: (num_protos, D)
            # 结果: (N_c, L, num_protos)
            sim = torch.einsum('nld,pd->nlp', class_local_feats, class_protos)
            
            # 对每个样本，聚合所有 patches 的原型激活
            # 方式: 对每个原型，取所有 patches 中的最大相似度 (max pooling)
            # 结果: (N_c, num_protos)
            max_sim_per_proto, _ = sim.max(dim=1)  # (N_c, num_protos)
            
            # 用 softmax 得到软分配分布
            soft_assign = F.softmax(max_sim_per_proto / temperature, dim=-1)  # (N_c, num_protos)
            
            # 计算类内平均分布 (作为 target)
            mean_dist = soft_assign.mean(dim=0, keepdim=True)  # (1, num_protos)
            
            # 计算每个样本与平均分布的 KL 散度
            # KL(p || q) = sum(p * log(p/q))
            # 使用 F.kl_div，注意输入是 log_softmax
            log_soft_assign = F.log_softmax(max_sim_per_proto / temperature, dim=-1)
            
            # KL 散度: 每个样本与平均分布的距离
            # F.kl_div 期望 input 是 log 概率，target 是概率
            kl_loss = F.kl_div(
                log_soft_assign,
                mean_dist.expand(N_c, -1),
                reduction='batchmean'
            )
            
            consistency_loss += kl_loss
            valid_classes += 1
        
        return consistency_loss / max(valid_classes, 1)
    
    def forward(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
        proto_indices: List[Tuple[int, int]],
        valid_mask: torch.Tensor,
        local_feats: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失 (优化版)
        
        损失组成:
        - cluster_loss: 拉近样本与同类原型
        - sep_loss: 推远样本与异类原型
        - soft_orth_loss: 类内原型软正交 (替代 div + contrastive)
        - local_consistency_loss: 同类样本原型激活一致性 (可选)
        
        Args:
            similarities: 相似度矩阵 (B, num_classes, max_prototypes)
            labels: 标签 (B,)
            prototypes: 所有原型
            proto_indices: 每类原型索引范围
            valid_mask: 有效原型掩码
            local_feats: 局部 patch 特征 (B, L, emb_dim)，可选
        """
        distances = 1 - similarities
        loss_dict = {}
        
        loss_dict['cluster_loss'] = self.compute_cluster_loss(
            distances, labels, proto_indices) * self.clst_scale
        loss_dict['sep_loss'] = self.compute_separation_loss(
            distances, labels, proto_indices) * self.sep_scale
        loss_dict['soft_orth_loss'] = self.compute_soft_orthogonality_loss(
            prototypes, proto_indices) * self.orth_scale
        
        # 局部一致性损失 (可选)
        if local_feats is not None and self.local_consistency_scale > 0:
            loss_dict['local_consistency_loss'] = self.compute_local_consistency_loss(
                local_feats, labels, prototypes, proto_indices,
                temperature=self.local_consistency_temp
            ) * self.local_consistency_scale
        
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
    """
    优化的动态原型管理器 - 标准 EMA 动量更新
    
    修复: 添加与 models_mae.py push_prototypes 兼容的 API:
    - var_threshold: 方差阈值
    - expand_prototypes (复数): 兼容旧 API
    - apply_momentum_updates: 批量应用更新
    """
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        init_prototypes: int = 5,
        max_prototypes: int = 10,
        momentum: float = 0.9,
        var_threshold: float = 0.5,  # 新增: 方差阈值
    ):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.max_prototypes = max_prototypes
        self.momentum = momentum
        self.var_threshold = var_threshold  # 新增: 用于 push 时判断是否扩展原型
        
        self.register_buffer('protos_per_class', torch.tensor([init_prototypes] * num_classes))
        
        total_max = num_classes * max_prototypes
        self.prototypes = nn.Parameter(torch.randn(total_max, emb_dim) * 0.01)
        nn.init.xavier_uniform_(self.prototypes)
        
        self.register_buffer('valid_mask', self._create_valid_mask())
        
        # 新增: 用于批量动量更新的缓存
        self._pending_updates: Dict[int, torch.Tensor] = {}
    
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
        """扩展单个原型 (单数形式)"""
        if self.protos_per_class[class_idx] >= self.max_prototypes:
            return False
        new_idx = class_idx * self.max_prototypes + self.protos_per_class[class_idx].item()
        self.prototypes.data[new_idx] = F.normalize(new_proto, p=2, dim=-1)
        self.protos_per_class[class_idx] += 1
        self.valid_mask[new_idx] = True
        return True
    
    @torch.no_grad()
    def expand_prototypes(self, class_idx: int, new_proto: torch.Tensor) -> bool:
        """
        扩展原型 (复数形式) - 兼容 models_mae.py 的 API
        实际上调用 expand_prototype
        """
        return self.expand_prototype(class_idx, new_proto)
    
    @torch.no_grad()
    def momentum_update(self, proto_idx: int, new_value: torch.Tensor):
        """
        标准 EMA 更新 - 缓存更新，等待 apply_momentum_updates 批量应用
        
        这样设计是为了兼容 models_mae.py 的调用模式:
        1. 多次调用 momentum_update 缓存更新
        2. 最后调用 apply_momentum_updates 批量应用
        """
        self._pending_updates[proto_idx] = new_value.clone()
    
    @torch.no_grad()
    def apply_momentum_updates(self):
        """
        批量应用所有缓存的动量更新
        兼容 models_mae.py 的 push_prototypes 调用模式
        """
        for proto_idx, new_value in self._pending_updates.items():
            new_value_norm = F.normalize(new_value, p=2, dim=-1)
            self.prototypes.data[proto_idx] = (
                self.momentum * self.prototypes.data[proto_idx] +
                (1 - self.momentum) * new_value_norm
            )
            self.prototypes.data[proto_idx] = F.normalize(
                self.prototypes.data[proto_idx], p=2, dim=-1
            )
        # 清空缓存
        self._pending_updates.clear()
    
    @torch.no_grad()
    def momentum_update_immediate(self, proto_idx: int, new_value: torch.Tensor):
        """立即应用动量更新 (不缓存)"""
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
    最优多原型分类头 (优化版)
    
    核心改进:
    1. 软正交损失替代 Contrastive + Diversity
    2. 三路门控融合
    3. 标准 EMA 动量更新
    4. 可选 Class-Balanced 加权
    5. 局部一致性损失 (同类样本原型激活一致性)
    
    损失组成:
    - L_ce: 交叉熵损失
    - L_cluster: 拉近样本与同类原型
    - L_separation: 推远样本与异类原型
    - L_soft_orth: 类内原型软正交
    - L_local_consistency: 同类样本原型激活一致性 (可选)
    
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
        sep_scale: float = 0.12,      # 提高 (原 0.08)
        orth_scale: float = 0.08,     # 新增: 软正交权重
        orth_threshold: float = 0.3,  # 新增: 软正交阈值
        local_consistency_scale: float = 0.05,  # 新增: 局部一致性权重
        local_consistency_temp: float = 0.1,    # 新增: 局部一致性温度
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
            orth_scale=orth_scale,
            orth_threshold=orth_threshold,
            local_consistency_scale=local_consistency_scale,
            local_consistency_temp=local_consistency_temp,
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
        """
        特征前向传播
        
        Args:
            pool_img_features: 局部 patch 特征 (B, L, C)，L=49 for 7x7 patches
            img_features: 全局特征 (B, C) 或 (B, 1, C)
        
        Returns:
            similarities: 相似度矩阵 (B, num_classes, max_prototypes)
            class_emb: 类别嵌入
            attn_weights: 注意力权重
            proj_img_features: 投影后的全局特征
            proj_local_feats: 投影后的局部特征 (用于局部一致性损失)
        """
        B, N, C = pool_img_features.shape
        
        if img_features.dim() == 3:
            img_features = img_features.mean(dim=1)
        
        img_features = self.proj(img_features)
        pool_img_features_proj = self.proj(pool_img_features)  # (B, L, emb_dim)
        pool_flat = pool_img_features_proj.reshape(1, B * N, self.emb_dim)
        
        prototypes = self.proto_manager.get_prototypes()
        proto_emb = prototypes.unsqueeze(0)
        
        class_emb, attn_weights = self.mixed_attn(proto_emb, pool_flat)
        ffn_out = self.ffn(class_emb)
        class_emb = self.ffn_norm(class_emb + ffn_out)
        class_emb = class_emb.expand(B, -1, -1)
        
        similarities = self.compute_similarities(img_features, class_emb)
        return similarities, class_emb, attn_weights, img_features, pool_img_features_proj
    
    def forward(
        self,
        pool_img_features: torch.Tensor,
        img_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        返回格式: (logits, loss, loss_dict)
        
        【修复】CB 加权现在同时作用于 CE Loss 和 Cluster Loss
        【新增】支持局部一致性损失
        """
        similarities, class_emb, _, proj_img_features, proj_local_feats = self.forward_features(
            pool_img_features, img_features
        )
        class_logits = self.aggregate_similarities(similarities)
        logits = class_logits * self.temperature
        
        if self.training and labels is not None:
            # 【修复】CE Loss 也使用 CB 加权
            ce_weights = self.proto_loss_fn.get_ce_weights(labels)
            if ce_weights is not None:
                ce_loss = F.cross_entropy(logits, labels, weight=ce_weights)
            else:
                ce_loss = F.cross_entropy(logits, labels)
            
            # 【新增】传递局部特征用于局部一致性损失
            proto_losses = self.proto_loss_fn(
                similarities=similarities,
                labels=labels,
                prototypes=self.proto_manager.prototypes,
                proto_indices=self.proto_manager.get_prototype_indices(),
                valid_mask=self.proto_manager.valid_mask,
                local_feats=proj_local_feats  # 新增: 传递投影后的局部特征
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
    local_consistency_scale: float = 0.05,
    local_consistency_temp: float = 0.1,
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
        local_consistency_scale: 局部一致性损失权重 (默认 0.05)
        local_consistency_temp: 局部一致性 softmax 温度 (默认 0.1)
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
        local_consistency_scale=local_consistency_scale,
        local_consistency_temp=local_consistency_temp,
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
    print("测试最优多原型分类头 (HeadOptimal) - 含局部一致性损失")
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
    logits, loss, loss_dict = main_model_head(x_fu1, x_fu, labels_7)
    print(f"  输入 pool_features: {x_fu1.shape}")
    print(f"  输入 img_features: {x_fu.shape}")
    print(f"  输出 Logits: {logits.shape}, Total Loss: {loss.item():.4f}")
    print("  损失分解:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"    {name}: {value.item():.4f}")
    print("  ✓ 与 main_model 完全兼容!")
    
    # 测试6: 局部一致性损失
    print("\n" + "-" * 50)
    print("测试6: 局部一致性损失 (Local Consistency Loss)")
    model_lc = HeadOptimal(
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_heads=num_heads,
        img_feat_dim=img_feat_dim,
        num_prototypes=5,
        max_prototypes=10,
        local_consistency_scale=0.05,  # 启用局部一致性
        local_consistency_temp=0.1,
    ).to(device)
    
    # 使用 49 patches (7x7) 模拟 main_model 的 x_fu1
    pool_49 = torch.randn(batch_size, 49, img_feat_dim).to(device)
    img_feat = torch.randn(batch_size, img_feat_dim).to(device)
    
    model_lc.train()
    logits, loss, loss_dict = model_lc(pool_49, img_feat, labels)
    print(f"  输入 pool_features: {pool_49.shape} (49 patches)")
    print(f"  Total Loss: {loss.item():.4f}")
    print("  损失分解:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"    {name}: {value.item():.4f}")
    
    if 'local_consistency_loss' in loss_dict:
        print("  ✓ 局部一致性损失已启用!")
    
    # 测试7: 检测函数
    print("\n" + "-" * 50)
    print("测试7: has_multi_prototype_head 检测函数")
    
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
