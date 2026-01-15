import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class Head(nn.Module):
#     def __init__(self, num_classes, emb_dim, num_heads, img_feat_dim):
#         super(Head, self).__init__()
#         self.num_classes = num_classes
#         self.emb_dim = emb_dim
#
#         # Embedding
#         self.class_embeddings = nn.Embedding(num_classes, emb_dim)
#
#         # Transformer
#         self.self_attn = nn.MultiheadAttention(emb_dim, num_heads,batch_first=True)
#         self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads,batch_first=True)
#
#         self.proj = nn.Linear(img_feat_dim, emb_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward_train(self, pool_img_features,img_features):
#         B, N, C = pool_img_features.shape
#         print(pool_img_features.shape)
#         img_features = self.proj(img_features)
#         print(img_features.shape)
#         pool_img_features = self.proj(pool_img_features)
#         pool_img_features = pool_img_features.reshape(B * N, C).unsqueeze(0)
#         print(pool_img_features.shape)
#         class_emb = self.class_embeddings.weight.reshape(1,self.num_classes,C)
#         print(class_emb.shape)
#         # Cross-attn
#         class_emb, _ = self.cross_attn(class_emb, pool_img_features, pool_img_features)
#         print(class_emb.shape)
#         # Self-attention
#         # class_emb, _ = self.self_attn(class_emb, class_emb, class_emb)
#         # print("Self-attention after", class_emb)
#         class_emb = class_emb.expand(B,-1,-1)
#         print(class_emb.shape)
#         # img_features = img_features.transpose(1,2)
#         #
#         #
#         # logits = torch.bmm(class_emb, img_features).squeeze(2)
#         logits = F.cosine_similarity(img_features, class_emb, dim=-1) * 10
#         # logits = F.cosine_similarity(img_features, class_emb, dim=-1) * 10
#
#         return logits
#
#     def forward(self,pool_img_features,img_features):
#         if self.training:
#             logits = self.forward_train(pool_img_features,img_features)
#             loss = self.compute_loss(self.class_embeddings)
#             return logits,loss
#         else:
#             logits = self.forward_train(pool_img_features,img_features)
#             return logits
#
#     def compute_loss(self, class_embeddings, contrastive_scale=0.1):
#
#         norms = class_embeddings.weight.norm(dim=1, p=2, keepdim=True)
#         dist_matrix = 1 - torch.matmul(class_embeddings.weight, class_embeddings.weight.t()) / norms[:, None] / norms[None, :]
#         dist_matrix = dist_matrix * (1 - torch.eye(dist_matrix.size(0), device=dist_matrix.device))
#         contrastive_loss = torch.sum(dist_matrix) * contrastive_scale
#
#         return contrastive_loss
class Head(nn.Module):
    def __init__(self, num_classes, emb_dim, num_heads, img_feat_dim, num_prototypes=5):
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_prototypes = num_prototypes  # Number of prototypes per class


        # Prototypes for each class
        self.class_prototypes = nn.Embedding(num_classes * num_prototypes, emb_dim)  # K prototypes per class

        # Transformer layers
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        # Linear projection for image features
        self.proj = nn.Linear(img_feat_dim, emb_dim)
        self.softmax = nn.Softmax(dim=-1)

        # FC layer to map from similarity scores to class logits
        self.fc = nn.Linear(num_prototypes, num_classes)

    def forward_train(self, pool_img_features, img_features):
        B, N, C = pool_img_features.shape
        img_features = self.proj(img_features)
        pool_img_features = self.proj(pool_img_features)
        print(pool_img_features.shape)
        pool_img_features = pool_img_features.reshape(B * N, C).unsqueeze(0)
        # pool_img_features = pool_img_features.expand(B * N, self.num_prototypes, C)
        print(pool_img_features.shape)
        # Generate the class prototypes
        class_emb = self.class_prototypes.weight.reshape(1,self.num_classes * self.num_prototypes, C)
        print(class_emb.shape)

        # Cross-attention
        class_emb, _ = self.cross_attn(class_emb, pool_img_features, pool_img_features)
        print(class_emb.shape)
        # Expand class_emb to match batch size
        class_emb = class_emb.reshape(1, self.num_classes, self.num_prototypes, self.emb_dim)
        class_emb = class_emb.expand(B, self.num_classes, self.num_prototypes, self.emb_dim)

        # Adjust img_features to match the shape of class_emb
        print(img_features.shape)
        img_features_exp = img_features.unsqueeze(1)  # Shape becomes (B, 1, 1, emb_dim)
        print(img_features_exp.shape)
        img_features_exp = img_features_exp.expand(B, self.num_classes, self.num_prototypes, self.emb_dim)
        print(img_features_exp.shape)
        # Calculate cosine similarity between class_emb and img_features
        cosine_sim = F.cosine_similarity(img_features_exp, class_emb, dim=-1)  # Shape: (B, num_classes, num_prototypes)
        print(cosine_sim.shape)
        # Map similarity scores to logits using a fully connected layer
        logits = self.fc(cosine_sim)  # Shape: (B, num_classes)
        print(logits.shape)

        # Compute prototype alignment loss (push loss)
        push_loss = self.compute_push_loss(class_emb)

        return logits, push_loss

    def forward(self, pool_img_features, img_features):
        if self.training:
            logits, push_loss = self.forward_train(pool_img_features, img_features)
            loss = self.compute_loss(self.class_prototypes) + push_loss
            return logits, loss
        else:
            logits = self.forward_train(pool_img_features, img_features)
            return logits

    def compute_loss(self, class_prototypes, contrastive_scale=0.1):
        # Contrastive loss computation (minimize distances between prototypes of the same class)
        norms = class_prototypes.weight.norm(dim=1, p=2, keepdim=True)
        dist_matrix = 1 - torch.matmul(class_prototypes.weight, class_prototypes.weight.t()) / norms[:, None] / norms[
                                                                                                                None, :]
        dist_matrix = dist_matrix * (1 - torch.eye(dist_matrix.size(0), device=dist_matrix.device))
        contrastive_loss = torch.sum(dist_matrix) * contrastive_scale
        return contrastive_loss

    def compute_push_loss(self, class_emb, scale=0.1):
        # Push loss: Minimize the distance between prototypes and training samples
        # Assuming we have the ground truth for each training sample, we can calculate the loss
        # For simplicity, we use cosine similarity as the distance measure.
        push_loss = torch.mean(torch.norm(class_emb, p=2, dim=-1))  # Simplified version of push loss
        return push_loss


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    x = torch.randn(32,1,384)
    x1 = torch.randn(32,196,384)
    y = nn.LayerNorm(384)
    x = y(x)
    net = Head(3,384,12,384)
    z = net.forward_train(x1,x)
    print(z)
