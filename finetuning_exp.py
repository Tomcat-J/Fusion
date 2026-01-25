# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
from projects.eval_tools.finetuning_mae_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=300):
        super(Exp, self).__init__(batch_size, max_epoch)

        # optimizer
        self.clip_grad = None
        # self.clip_mode = "norm"

        # augmentation & regularization
        self.color_jitter = 0.3
        self.aa = "rand-m10-mstd0.5-inc1"
        self.reprob = 0.0
        self.mixup = 0.0
        self.cutmix = 0.0
        self.smoothing = 0.0
        self.drop_path = 0.0
        
        # ===== MHF Enhancement Settings (Task 12) =====
        # Requirements: 6.3
        self.use_mhf_enhancement = True  # 是否启用MHF增强
        self.enhancement_reduction = 0.0625  # 增强模块通道缩减比例 (1/16)
        
        # 差异化学习率设置
        # Requirements: 5.3, 5.4
        self.backbone_lr_scale = 0.1  # backbone学习率倍率
        self.mhf_lr_scale = 0.5  # MHF原始参数学习率倍率
        self.enhancement_lr_scale = 1.0  # 增强模块学习率倍率
        
        # 渐进式训练设置
        self.freeze_backbone_epochs = 0  # 冻结backbone的epoch数

        # self.num_workers = 10
        self.weights_prefix = ""
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = "ft_"


if __name__ == "__main__":
    exp = Exp(2)
    model = exp.get_model()
    # print(model)
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)
        if param.grad is not None:
            print(f"{name} gradient: {param.grad}")
    loader = exp.get_data_loader()
    opt = exp.get_optimizer()
    scheduler = exp.get_lr_scheduler()
