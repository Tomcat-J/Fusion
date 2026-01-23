import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
import numpy as np
from ._base import Distiller


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


class Two(Distiller):
    def __init__(self, student, teacher1, teacher2,cfg):
        super(Two, self).__init__(student, teacher1,teacher2)
        self.shapes1 = cfg.Two.SHAPES1
        self.shapes2 = cfg.Two.SHAPES2
        self.out_shapes1 = cfg.Two.OUT_SHAPES1
        self.out_shapes2 = cfg.Two.OUT_SHAPES2
        in_channels1 = cfg.Two.IN_CHANNELS1
        in_channels2 = cfg.Two.IN_CHANNELS2
        out_channels1 = cfg.Two.OUT_CHANNELS1
        out_channels2 = cfg.Two.OUT_CHANNELS2
        self.ce_loss_weight1 = cfg.Two.CE_WEIGHT1
        self.reviewkd_loss_weight = cfg.Two.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.Two.WARMUP_EPOCHS
        self.stu_preact = cfg.Two.STU_PREACT
        self.max_mid_channel = cfg.Two.MAX_MID_CHANNEL

        self.ce_loss_weight2 = cfg.Two.CE_WEIGHT2
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

        abfs1 = nn.ModuleList()
        abfs2 = nn.ModuleList()
        mid_channel = min(512, in_channels1[-1])
        for idx, in_channel in enumerate(in_channels1):
            abfs1.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels1[idx],
                    idx < len(in_channels1) - 1,
                )
            )
        self.abfs1 = abfs1[::-1]
        for idx, in_channel in enumerate(in_channels2):
            abfs2.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels2[idx],
                    idx < len(in_channels2) - 1,
                )
            )
        self.abfs2 = abfs2[::-1]


    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()] + list(self.abfs1.parameters())+ list(self.abfs2.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs1.parameters():
            num_p += p.numel()
        for p in self.abfs2.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher1, features_teacher1 = self.teacher1(image)
            logits_teacher2, features_teacher2 = self.teacher2(image)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ] #[f0,f1,f2,f3,avg]
        x = x[::-1] #翻转


        results1 = []
        out_features1, res_features1 = self.abfs1[0](x[0], out_shape=self.out_shapes1[0])
        results1.append(out_features1)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs1[1:], self.shapes1[1:], self.out_shapes1[1:]
        ):
            out_features, res_features1 = abf(features, res_features1, shape, out_shape)
            results1.insert(0, out_features)
        '''
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        '''
        features_teacher1 = features_teacher1["feats"][1:] + [
            features_teacher1["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]#[f1,f2,f3,avg]


        results2 = []
        out_features2, res_features2 = self.abfs2[0](x[0], out_shape=self.out_shapes2[0])
        results2.append(out_features2)
        for features, abf, shape, out_shape in zip(
                x[1:], self.abfs2[1:], self.shapes2[1:], self.out_shapes2[1:]
        ):
            out_features, res_features2 = abf(features, res_features2, shape, out_shape)
            results2.insert(0, out_features)
        '''
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        '''
        features_teacher2 = features_teacher2["feats"][1:] + [
            features_teacher2["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]  # [f1,f2,f3,avg]


        # losses

        epoch = kwargs.get('epoch', None)
        beta = 0.9999
        gamma = 0.5
        loss_type = "focal"
        samples_per_cls = [41, 1317, 196, 121, 546]
        no_of_classes = 5
        loss_ce = (0.5 + epoch / 200) * CB_loss(target, logits_student, samples_per_cls, no_of_classes, loss_type, beta,
                                             gamma) + (0.5 - epoch / 200) * F.cross_entropy(logits_student, target)

        CE1 = F.cross_entropy(logits_teacher1, target)
        CE2 = F.cross_entropy(logits_teacher2, target)
        CE = CE1+CE2

        loss_reviewkd1 = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * hcl_loss(results1, features_teacher1)
        )


        loss_reviewkd2 = (
                self.reviewkd_loss_weight
                * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
                * hcl_loss(results2, features_teacher2)
        )


        loss_dkd1 = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher1,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )


        loss_dkd2 = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher2,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )


        loss_reviewkd = (1-CE1/CE)*loss_reviewkd1+(1-CE2/CE)*loss_reviewkd2
        loss_dkd = (1-CE1/CE)*loss_dkd1+(1-CE2/CE)*loss_dkd2
        loss_kd = loss_reviewkd+loss_dkd

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        '''
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        '''
        if fuse:
            self.att_conv = SpatialAttention()
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x

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

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

