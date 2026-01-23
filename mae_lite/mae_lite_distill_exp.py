# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os, sys
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
import numpy as np
from timm.models import create_model
import torch.nn.functional as F
from projects.mae_lite.mae_lite_exp import Exp as BaseExp, MAE
from loguru import logger
from projects.eval_tools.util.forward_hook import AttnCatcher, RepCatcher


class MAE_distill(MAE):
    def __init__(self, args, model, tch_model,aux_model):
        super(MAE_distill, self).__init__(args, model)
        self.tch_model = tch_model
        self.aux_model = aux_model
        for p in self.tch_model.parameters():
            p.requires_grad = False
        for p in self.aux_model.parameters():
            p.requires_grad = False
        self.tch_model.eval()
        self.aux_model.eval()
        self.distill_criterion = MSELoss()
        # attn
        self.distill_attn_idx = args.distill_attn_idx
        if self.distill_attn_idx is not None and len(self.distill_attn_idx) > 0:
            self.attn_alpha = args.distill_attn_alpha
            self.attn_catcher = AttnCatcher(self.model, self.distill_attn_idx)
            # TODO: align teacher and student layers when the depths are different.
            tch_distill_attn_idx = self.distill_attn_idx
            self.tch_attn_catcher = AttnCatcher(self.tch_model, tch_distill_attn_idx)
            use_attn_adapter = args.use_attn_adapter
            if self.model.num_heads != self.tch_model.num_heads:
                use_attn_adapter = True
            if use_attn_adapter:
                self.attn_adapter = nn.ModuleDict(
                    {
                        "adapter{}".format(idx-1): nn.Conv2d(self.model.num_heads, self.tch_model.num_heads, 1, bias=False)
                        for idx in self.distill_attn_idx
                    }
                )
            else:
                self.attn_adapter = nn.ModuleDict({"adapter{}".format(idx-1): nn.Identity() for idx in self.distill_attn_idx})
        # hidden
        self.distill_hidden_idx = args.distill_hidden_idx
        if self.distill_hidden_idx is not None and len(self.distill_hidden_idx) > 0:
            self.hidden_alpha = args.distill_hidden_alpha
            self.hidden_catcher = RepCatcher(self.model, self.distill_hidden_idx)
            # TODO: align teacher and student layers when the depths are different.
            tch_distill_hidden_idx = self.distill_hidden_idx
            self.tch_hidden_catcher = RepCatcher(self.tch_model, tch_distill_hidden_idx)
            use_hidden_adapter = args.use_hidden_adapter
            if self.model.embed_dim != self.tch_model.embed_dim:
                use_hidden_adapter = True
            if use_hidden_adapter:
                self.hidden_adapter = nn.ModuleDict(
                    {
                        "adapter{}".format(idx): nn.Linear(self.model.embed_dim, self.tch_model.embed_dim, bias=False)
                        for idx in self.distill_hidden_idx
                    }
                )
            else:
                self.hidden_adapter = nn.ModuleDict({"adapter{}".format(idx): nn.Identity() for idx in self.distill_hidden_idx})

        #------------------------
        self.ce_loss_weight = 1  # 1
        self.reviewkd_loss_weight = 0.2
        self.warmup_epochs = 20
        self.shapes = [14, 14, 14, 14]
        self.out_shapes = [1,7,14,28]
        in_channels = [384, 384, 384, 384]
        out_channels = [ 256, 512, 1024,1024]
        self.max_mid_channe = 768
        self.stu_preact = False
        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def train(self, mode=True):
        super().train(mode)
        self.tch_model.eval()

    def get_distill_attn_loss(self):
        loss_attn = 0
        if len(self.distill_attn_idx) > 0:
            for idx in self.distill_attn_idx:
                adapter = self.attn_adapter["adapter{}".format(idx-1)]
                attn = self.attn_catcher.get_features(idx)
                tch_attn = self.tch_attn_catcher.get_features(idx).detach()
                loss_attn += self.distill_criterion(adapter(attn), tch_attn)
        return loss_attn
    
    def get_distill_hidden_loss(self):
        loss_hidden = 0
        if len(self.distill_hidden_idx) > 0:
            for idx in self.distill_hidden_idx:
                adapter = self.hidden_adapter["adapter{}".format(idx)]
                hidden = self.hidden_catcher.get_features(idx, remove_cls_token=False)
                tch_hidden = self.tch_hidden_catcher.get_features(idx, remove_cls_token=False).detach()
                loss_hidden += self.distill_criterion(adapter(hidden), tch_hidden)
        return loss_hidden



    def forward(self, x, target=None, update_param=False,epoch = None,temp = 1.0,temp_mlp = None,cos_value = None):
        if self.training:
            images = x
            outcomes_student,features_student = self.model.forward_features(images)
            logits_student = self.model.head(outcomes_student)
            with torch.no_grad():
                logits_teacher,features_teacher = self.aux_model(images)
                # outcomes_teacher,features_teacher = self.tch_model.forward_features(images)
                # logits_teacher = self.tch_model.head(outcomes_teacher)

            if self.mixup_fn is not None:
                images, _ = self.mixup_fn(images, target)
            loss, _, _, ids_shuffle = self.model(images, self.mask_ratio, None)

            with torch.no_grad():
                _, _, _, _ = self.tch_model.forward_encoder(
                    images, self.mask_ratio, ids_shuffle
                )
            temp = temp_mlp(logits_student, logits_teacher, cos_value)  # (teacher_output, student_output)
            temp = 1 + 20 * torch.sigmoid(temp)
            temp = temp.cuda()
            output_dict = {}
            loss_attn = self.get_distill_attn_loss()
            if isinstance(loss_attn, torch.Tensor):
                output_dict["attn"] = loss_attn.detach().item()
                loss += loss_attn * self.attn_alpha
            loss_hidden = self.get_distill_hidden_loss()
            if isinstance(loss_hidden, torch.Tensor):
                output_dict['hidden'] = loss_hidden.detach().item()
                loss += loss_hidden * self.hidden_alpha

            # get features
            if self.stu_preact:
                xx = features_student["preact_feats"] + [
                    features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
                ]
            else:
                xx = features_student["feats"] + [
                    features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
                ]
            xx = xx[::-1]
            results = []
            out_features, res_features = self.abfs[0](xx[0], out_shape=self.out_shapes[0])
            results.append(out_features)
            for features, abf, shape, out_shape in zip(
                    xx[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
            ):
                out_features, res_features = abf(features, res_features, shape, out_shape)
                results.insert(0, out_features)
            features_teacher = features_teacher["feats"][1:] + [
                features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
            # losses
            loss_reviewkd = (
                    self.reviewkd_loss_weight
                    * min(epoch / self.warmup_epochs, 1.0)
                    * hcl_loss(results, features_teacher)
            )


            beta = 0.9999
            gamma_ce = 0.5
            gamma_nkd = 1.5
            gamma_rekd = 2
            beta_rekd = 8
            gamma_dkd = 1
            beta_dkd = 8
            loss_type = "focal"
            samples_per_cls = [2408, 1833,2158]
            no_of_classes = 3
            loss_ce = (0.5 + epoch / 200) * CB_loss(target, logits_student, samples_per_cls, no_of_classes, loss_type,
                                                    beta,
                                                    gamma_ce) + (0.5 - epoch / 200) * F.cross_entropy(logits_student,target)
            # loss_nkd = nkd_loss(logits_student, logits_teacher, target, gamma_nkd, temp)
            # loss_rekd = rekd_loss(logits_student,logits_teacher,gamma_rekd,beta_rekd,temp)
            loss_dkd = dkd_loss(logits_student,logits_teacher,target,gamma_dkd,beta_dkd,temp)
            if self.ema_model is not None:
                self.ema_model.update(self.model)
            losses_dict = {
                "loss_ce": loss_ce,
                # "loss_nkd":loss_nkd,
                # "loss_rekd":loss_rekd,
                "loss_dkd":loss_dkd,
                "loss_reviewkd":loss_reviewkd,
                "loss": loss,
            }
            return losses_dict, output_dict,temp
        else:
            logits = self.model(x)
            return logits


def set_model_weights(model, ckpt_path, weights_prefix):
    BLACK_LIST = ("head",)

    def _match(key):
        return any([k in key for k in BLACK_LIST])

    if not os.path.isfile(ckpt_path):
        from torch.nn.modules.module import _IncompatibleKeys

        logger.info("No checkpoints found! Training from scratch!")
        return _IncompatibleKeys(missing_keys=None, unexpected_keys=None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not weights_prefix:
        state_dict = { k: v for k, v in ckpt["model"].items()}
    else:
        if weights_prefix and not weights_prefix.endswith("."):
            weights_prefix += "."
        if all(key.startswith("module.") for key in ckpt["model"].keys()):
            weights_prefix = "module." + weights_prefix
        state_dict = {k.replace(weights_prefix, ""): v for k, v in ckpt["model"].items() if not _match(k)}

    msg = model.load_state_dict(state_dict, strict=False)
    del state_dict
    return msg


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)
        self.encoder_arch = "mae_vit_small_patch16"
        self.mask_ratio = 0.75
        self.distill_attn_alpha = 1.0
        self.distill_attn_idx = (12,)  # 1-12
        self.use_attn_adapter = True
        self.distill_hidden_alpha = 1.0
        self.distill_hidden_idx = ()  # 0-12
        self.use_hidden_adapter = True

        self.aux_teacher_arch = "convnextv2_base"
        self.teacher_arch = "mae_vit_base_patch16"
        self.teacher_weights_prefix = ""
        self.aux_teacher_weights_prefix = ""
        self.student_weights_prefix = "model"
        self.teacher_ckpt_path = "/home/backup/lh/KD/projects/mae_lite/checkpoints/mae_base_1600e.pth.tar"
        self.aux_teacher_ckpt_path = "/home/backup/lh/KD/projects/mae_lite/checkpoints/convnextv2_base_22k_224_ema.pt"
        self.student_ckpt_path = "/home/backup/lh/KD/projects/mae_lite/checkpoints/mae_small_distill_400e.pth.tar"
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]
        # self.exp_name = "mae_lite/mae_small_distill_400e"
    def get_model(self):
        if "model" not in self.__dict__:
            model = create_model(self.encoder_arch, norm_pix_loss=self.norm_pix_loss)
            tch_model = create_model(self.teacher_arch, norm_pix_loss=self.norm_pix_loss)
            aux_model = create_model(self.aux_teacher_arch)
            del tch_model.mask_token
            del tch_model.decoder_embed
            del tch_model.decoder_pos_embed
            del tch_model.decoder_blocks
            del tch_model.decoder_norm
            del tch_model.decoder_pred
            msg = set_model_weights(tch_model, self.teacher_ckpt_path, self.teacher_weights_prefix)
            msg2 = set_model_weights(aux_model, self.aux_teacher_ckpt_path, self.aux_teacher_weights_prefix)
            msg1 = set_model_weights(model, self.student_ckpt_path, self.student_weights_prefix)
            logger.info("Tch_Model params {} are not loaded".format(msg.missing_keys))
            logger.info("State-dict params {} are not used".format(msg.unexpected_keys))
            logger.info("Aux_Model params {} are not loaded".format(msg2.missing_keys))
            logger.info("State-dict params {} are not used".format(msg2.unexpected_keys))
            logger.info("Model params {} are not loaded".format(msg1.missing_keys))
            logger.info("State-dict params {} are not used".format(msg1.unexpected_keys))

            self.model = MAE_distill(self, model, tch_model,aux_model)
        return self.model


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

def nkd_loss(logit_s, logit_t, gt_label,temp = None,gamma = None):

    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)
    log_softmax = nn.LogSoftmax(dim=1)
    # N*class
    N, c = logit_s.shape
    s_i = log_softmax(logit_s)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = log_softmax(logit_s / temp)
    T_i = F.softmax(logit_t / temp, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = - gamma * (temp ** 2) * loss_non

    return loss_t + loss_non

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
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
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

# def uskd_loss()

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    logits_student = normalize(logits_student)
    logits_teacher = normalize(logits_teacher)

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


def rekd_loss(logits_student,logits_teacher,alpha = 0.8,beta = 0.2,T = 1.0,k = 1):
    # logits_student = normalize(logits_student)
    # logits_teacher = normalize(logits_teacher)

    s_mask = top_k_mask(logits_student,k)
    s_topk_logit = logits_student/T - 1000.0*s_mask
    t_topk_logit = logits_teacher/T - 1000.0*s_mask

    s_topk_logit = F.log_softmax(s_topk_logit,dim = 1)
    t_topk_logit = F.softmax(t_topk_logit,dim = 1)
    L_TISD = F.kl_div(s_topk_logit,t_topk_logit,size_average=False) * (T * T)/logits_student.size()[0]

    not_s_mask = not_topk_mask(logits_student,k)
    s_not_topk_logit = logits_student/T - 1000.0*not_s_mask
    t_not_topk_logit = logits_teacher/T - 1000.0*not_s_mask

    s_logsoftmax = F.log_softmax(s_not_topk_logit,dim = 1)
    t_softmax = F.softmax(t_not_topk_logit,dim = 1)
    L_NTID = F.kl_div(s_logsoftmax,t_softmax,size_average=False) * (T * T)/logits_student.size()[0]

    TotalKD = alpha * L_TISD + beta * L_NTID
    return TotalKD

def top_k_mask(logits, k):
    # Get the top-k indices
    topk_values, topk_indices = torch.topk(logits, k, dim=1)

    # Create a mask of the same shape as logits and set to False
    mask = torch.zeros_like(logits, dtype=torch.bool)

    # Place True in the positions of the top-k elements
    mask.scatter_(1, topk_indices, True)
    return mask

def not_topk_mask(logits, k):
    # Get the top-k indices
    topk_values, topk_indices = torch.topk(logits, k, dim=1)

    # Create a mask of the same shape as logits and set to True
    mask = torch.ones_like(logits, dtype=torch.bool)

    # Place False in the positions of the top-k elements
    mask.scatter_(1, topk_indices, False)
    return mask

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


if __name__ == "__main__":
    exp = Exp(2, 1)
    exp.num_workers = 1
    print(exp.exp_name)
    model = exp.get_model()
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
    train_loader = loader["train"]
    for inps in train_loader:
        images, target = inps
        out = model(images, target=target)
