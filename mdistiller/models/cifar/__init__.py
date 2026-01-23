import os
from .resnet import (
    resnet8,
    resnet14,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet8x4,
    resnet32x4,
)
from .resnetv2 import ResNet50, ResNet18
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .mv2_tinyimagenet import mobilenetv2_tinyimagenet
from .res2net import res2next29_6cx24wx6scale_se,res2next29_6cx24wx6scale_se_min
from .swimtransformer import swin_small_patch4_window7_224
from .convnext import convnext_base
from .coatnet import coatnet_2
from .convnext_v2 import convnextv2_base,convnextv2_tiny,convnextv2_pico,convnextv2_nano
from .swimtransformerv2 import swinv2_small_window16_256
from .vmamba import vmamba_small,vmamba_base
from .efficientv2 import efficientnetv2_m
from .mobilenetV3 import mobilenet_v3_large
from .VIT import vit_base_patch32_224_in21k

cifar100_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/cifar_teachers/"
)

tiny_imagenet_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/tiny_imagenet_teachers/"
)

cifar_model_dict = {
    # teachers
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.checkpoints",
    ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.checkpoints",
    ),
    "resnet32x4": (
        resnet32x4,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.checkpoints",
    ),
    "ResNet50": (
        ResNet50,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.checkpoints",
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.checkpoints",
    ),
    "vgg13": (vgg13_bn, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.checkpoints"),
    "res2next29_6cx24wx6scale_se":(res2next29_6cx24wx6scale_se,"/home/data/pyl/models/res2next29_6cx24wx6scale_se/model.checkpoints"),
    "convnext_v2_base": (convnextv2_base, ""),
    "swimtransformerv2":(swinv2_small_window16_256,"/home/data/pyl/test_result/lung_baselines_vanilla_swimtransformerv2_kd3_test/student_latest"),
    "vmamba":(vmamba_base,"/home/data/pyl/test_result/lung_baselines_vanilla_vmamba_kd3_test/student_latest"),
    # students
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet32": (resnet32, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_40_1": (wrn_40_1, None),
    "vgg8": (vgg8_bn, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "MobileNetV2": (mobile_half, None),
    "ShuffleV1": (ShuffleV1, None),
    "ShuffleV2": (ShuffleV2, None),
    "res2next29_6cx24wx6scale_se_min":(res2next29_6cx24wx6scale_se_min,None),
    "convnext_v2_tiny": (convnextv2_tiny, "/home/data/pyl/models/convnext_v2/convnextv2_tiny_1k_224_ema.pt"),
    "convnext_v2_tinys": (convnextv2_tiny, "/home/data/pyl/models/convnext_v2/convnextv2_tiny_1k_224_ema.pt"),
}


tiny_imagenet_model_dict = {
    "ResNet18": (ResNet18, tiny_imagenet_model_prefix + "ResNet18_vanilla/ti_res18"),
    "MobileNetV2": (mobilenetv2_tinyimagenet, None),
    "ShuffleV2": (ShuffleV2, None),
}

lung_model_dict = {
    #students
    "swimtransformer":(swin_small_patch4_window7_224,None),
    "convnext":(convnext_base,),
    "coatnet":(coatnet_2,None),
    "convnext_v2":(convnextv2_base,"/home/data/pyl/models/convnext_v2/convnextv2_base_1k_224_ema.pt"),
    "swimtransformerv2":(swinv2_small_window16_256,"/home/data/pyl/models/swimtranformerv2/swinv2_small_patch4_window16_256.checkpoints"),
    "convnext_v2_tiny": (convnextv2_tiny, "/home/data/pyl/models/convnext_v2/convnextv2_tiny_1k_224_ema.pt"),
    "convnextv2_nano":(convnextv2_nano,"/home/data/pyl/models/convnext_v2/convnextv2_nano_1k_224_ema.pt"),
    "convnextv2_pico": (convnextv2_pico, "/home/data/pyl/models/convnext_v2/convnextv2_pico_1k_224_ema.pt"),
    "vmamba":(vmamba_base,"/home/data/pyl/models/vmamba/vssm_base_0229_ckpt_epoch_237.checkpoints"),
    "ResNet50": (ResNet50, "/home/pyl/konwledage dis/mdistiller-master/download_ckpts/cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.checkpoints"),
    "efficinet":(efficientnetv2_m,"/home/data/pyl/models/efficinet/pre_efficientnetv2-m.checkpoints"),
    "mobilenet":(mobilenet_v3_large,"/home/data/pyl/models/mobilenet/mobilenet_v3_large-8738ca79.checkpoints"),
    "vit":(vit_base_patch32_224_in21k,"/home/data/pyl/models/VIT/jx_vit_base_patch32_224_in21k-8db57226.checkpoints")
}