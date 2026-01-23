import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, num_classes, emb_dim, num_heads, img_feat_dim):
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        # Embedding
        self.class_embeddings = nn.Embedding(num_classes, emb_dim)
        self.class_embeddings = self.class_embeddings.weight.unsqueeze(0)
        # Transformer
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads)

        self.proj = nn.Linear(img_feat_dim, emb_dim)

    def forward_train(self, img_features):
        img_features = self.proj(img_features)

        # Self-attention
        class_emb, _ = self.self_attn(self.class_embeddings, self.class_embeddings, self.class_embeddings)

        # Cross-attn
        updated_features, _ = self.cross_attn(class_emb, img_features, img_features)

        logits = F.cosine_similarity(updated_features, class_emb, dim=-1)

        return logits

    def forward(self,img_features,epoch,targets):
        if self.training:
            logits = self.forward_train(img_features)
            loss = self.compute_loss(logits,epoch,targets,self.class_embeddings)
            return logits,loss
        else:
            logits = self.forward_train(img_features)
            return logits

    def compute_loss(logits, epoch ,targets, class_embeddings, contrastive_scale=0.1):

        beta = 0.9999
        gamma_ce = 0.5
        loss_type = "focal"
        samples_per_cls = [2408, 1833, 2158]
        no_of_classes = 3
        loss_ce = (0.5 + epoch / 200) * CB_loss(targets, samples_per_cls, no_of_classes, loss_type,
                                                beta,
                                                gamma_ce) + (0.5 - epoch / 200) * F.cross_entropy(logits,targets)
        norms = class_embeddings.norm(dim=1)
        dist_matrix = torch.matmul(class_embeddings, class_embeddings.t()) / norms[:, None] / norms[None, :]
        dist_matrix = dist_matrix * (1 - torch.eye(dist_matrix.size(0), device=dist_matrix.device))
        contrastive_loss = -torch.sum(dist_matrix) * contrastive_scale

        return loss_ce + contrastive_loss

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 0.5
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    entroy_weights = weights
    entroy_weights = torch.tensor(entroy_weights).float()
    entroy_weights = entroy_weights.cuda(non_blocking=True)
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.cuda(non_blocking=True)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    elif loss_type == "entroy_cross":
        cb_loss = F.cross_entropy(logits, labels, weight=entroy_weights)
    return cb_loss

