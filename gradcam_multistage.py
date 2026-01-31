#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多阶段 Grad-CAM 热力图可视化工具
用于生成类似论文中 Stage1-4 的热力图对比图

使用方法:
python gradcam_multistage.py --model_path /path/to/model.pth --image_path /path/to/image.jpg --output_dir ./cam_results
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from projects.mae_lite import models_mae

matplotlib.use('Agg')  # 非交互式后端

import torch
import torch.nn.functional as F
from torchvision import transforms

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'projects', 'mae_lite'))


class GradCAM:
    """
    Grad-CAM实现类
    用于生成类激活映射热力图
    """
    
    def __init__(self, model, target_layers):
        """
        Args:
            model: PyTorch模型
            target_layers: 目标层列表，用于提取特征图
        """
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        for name, layer in self.target_layers.items():
            # 前向钩子
            def make_forward_hook(layer_name):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    self.activations[layer_name] = out.detach()
                return hook
            
            # 反向钩子
            def make_backward_hook(layer_name):
                def hook(module, grad_input, grad_output):
                    if isinstance(grad_output, tuple):
                        grad = grad_output[0]
                    else:
                        grad = grad_output
                    if grad is not None:
                        self.gradients[layer_name] = grad.detach()
                return hook
            
            fwd_handle = layer.register_forward_hook(make_forward_hook(name))
            bwd_handle = layer.register_full_backward_hook(make_backward_hook(name))
            self.handles.append(fwd_handle)
            self.handles.append(bwd_handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def __call__(self, input_tensor, target_class=None):
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量 (1, C, H, W)
            target_class: 目标类别索引，None则使用预测类别
            
        Returns:
            cams: 各层的CAM热力图字典
            pred_class: 预测类别
        """
        self.activations = {}
        self.gradients = {}
        
        # 前向传播
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # 获取预测类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 生成各层的CAM
        cams = {}
        for name in self.target_layers.keys():
            if name not in self.activations:
                print(f"警告: {name} 没有激活值")
                continue
            if name not in self.gradients:
                print(f"警告: {name} 没有梯度")
                continue
                
            activation = self.activations[name]
            gradient = self.gradients[name]
            
            # 处理不同形状的特征图
            if len(activation.shape) == 3:  # (B, N, C) - Transformer格式
                B, N, C = activation.shape
                H = W = int(np.sqrt(N - 1)) if N > 1 else int(np.sqrt(N))
                if N > H * W:
                    activation = activation[:, 1:, :]
                    gradient = gradient[:, 1:, :]
                activation = activation.permute(0, 2, 1).reshape(B, C, H, W)
                gradient = gradient.permute(0, 2, 1).reshape(B, C, H, W)
            
            # 全局平均池化梯度
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            
            # 加权求和
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # 归一化
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cams[name] = cam.squeeze().cpu().numpy()
        
        return cams, target_class


def get_hifuse_target_layers(model):
    """
    获取HiFuse模型的目标层
    返回4个Stage的MHF融合模块 (fu1-fu4)
    """
    target_layers = {}
    
    # 尝试获取模型的backbone
    if hasattr(model, 'model'):
        backbone = model.model
        if hasattr(backbone, 'model') and not hasattr(backbone, 'fu1'):
            backbone = backbone.model
    else:
        backbone = model
    
    print(f"  [get_hifuse_target_layers] backbone 类型: {type(backbone).__name__}")
    
    # 使用 MHF 融合模块 (fu1-fu4)
    if hasattr(backbone, 'fu1'):
        target_layers['Stage 1'] = backbone.fu1
        print(f"    找到 fu1: {type(backbone.fu1).__name__}")
    
    if hasattr(backbone, 'fu2'):
        target_layers['Stage 2'] = backbone.fu2
        print(f"    找到 fu2: {type(backbone.fu2).__name__}")
            
    if hasattr(backbone, 'fu3'):
        target_layers['Stage 3'] = backbone.fu3
        print(f"    找到 fu3: {type(backbone.fu3).__name__}")
            
    if hasattr(backbone, 'fu4'):
        target_layers['Stage 4'] = backbone.fu4
        print(f"    找到 fu4: {type(backbone.fu4).__name__}")
    
    # 如果没有 fu 层，使用 ConvNeXt 分支的 stages
    if not target_layers and hasattr(backbone, 'stages'):
        print("    使用 ConvNeXt stages 作为目标层")
        for i, stage in enumerate(backbone.stages[:4]):
            target_layers[f'Stage {i+1}'] = stage
    
    # 尝试 Swin Transformer 分支的 layers
    if not target_layers:
        if hasattr(backbone, 'layers1'):
            target_layers['Stage 1'] = backbone.layers1
        if hasattr(backbone, 'layers2'):
            target_layers['Stage 2'] = backbone.layers2
        if hasattr(backbone, 'layers3'):
            target_layers['Stage 3'] = backbone.layers3
        if hasattr(backbone, 'layers4'):
            target_layers['Stage 4'] = backbone.layers4
    
    # 通用 ResNet 风格的 layer
    if not target_layers:
        if hasattr(backbone, 'layer1'):
            target_layers['Stage 1'] = backbone.layer1
        if hasattr(backbone, 'layer2'):
            target_layers['Stage 2'] = backbone.layer2
        if hasattr(backbone, 'layer3'):
            target_layers['Stage 3'] = backbone.layer3
        if hasattr(backbone, 'layer4'):
            target_layers['Stage 4'] = backbone.layer4
    
    return target_layers


def visualize_cam(original_img, cams, save_path, title="Grad-CAM Visualization"):
    """
    可视化CAM热力图
    
    Args:
        original_img: 原始图像 (H, W, 3) numpy数组
        cams: CAM字典 {stage_name: cam_array}
        save_path: 保存路径
        title: 图像标题
    """
    n_stages = len(cams)
    fig, axes = plt.subplots(1, n_stages + 1, figsize=(4 * (n_stages + 1), 4))
    
    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # 各Stage的热力图
    for idx, (stage_name, cam) in enumerate(cams.items()):
        ax = axes[idx + 1]
        
        # 调整CAM大小
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # 叠加热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 混合原图和热力图
        superimposed = heatmap * 0.4 + original_img * 0.6
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        ax.imshow(superimposed)
        ax.set_title(stage_name, fontsize=12)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_figure(images_data, save_path, row_labels=None):
    """
    创建多行对比图（类似论文中的图5.2）
    
    Args:
        images_data: 列表，每个元素是 (original_img, cams_dict) 元组
        save_path: 保存路径
        row_labels: 每行的标签列表
    """
    n_rows = len(images_data)
    n_cols = 5  # 原图 + 4个Stage
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 列标题
    col_titles = ['Original', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    for row_idx, (original_img, cams) in enumerate(images_data):
        # 原始图像
        axes[row_idx, 0].imshow(original_img)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')
        
        # 行标签
        if row_labels and row_idx < len(row_labels):
            axes[row_idx, 0].set_ylabel(row_labels[row_idx], fontsize=12, fontweight='bold', rotation=0, labelpad=50, va='center')
        
        # 各Stage热力图
        stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        for col_idx, stage_name in enumerate(stage_names):
            ax = axes[row_idx, col_idx + 1]
            
            if stage_name in cams:
                cam = cams[stage_name]
                cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
                
                # 生成热力图
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # 混合
                superimposed = heatmap * 0.4 + original_img * 0.6
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                
                ax.imshow(superimposed)
            else:
                ax.imshow(original_img)
            
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx + 1], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比图已保存到: {save_path}")


def load_model(model_path, num_classes=7, device='cuda', use_enhanced=False, enhancement_reduction=0.0625):
    """
    加载模型 - 【关键修复】使用与 eval.py 完全相同的方式加载模型
    
    问题根因：
    - eval.py 使用 Model 类包装器，checkpoint 的 keys 是 "model.xxx"
    - 之前的实现直接创建裸模型，导致 key 不匹配
    
    解决方案：
    - 使用 finetuning_exp.Exp 类来创建模型（与 eval.py 相同）
    - 使用 set_model_weights() 加载权重（与 eval.py 相同）
    - 返回内部的 encoder 模型用于 GradCAM（因为 GradCAM 需要访问中间层）
    
    Args:
        model_path: 模型权重路径
        num_classes: 类别数
        device: 设备
        use_enhanced: 是否使用增强版模型 (main_model_enhanced)
        enhancement_reduction: 增强模块的reduction参数
    """
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'projects', 'mae_lite'))
    
    from projects.mae_lite.mae_lite_distill_exp import set_model_weights
    
    # 确保 models_mae 中的模型已注册到 timm
    import models_mae  # 这会触发 @register_model 装饰器
    
    print("=" * 60)
    print("【GradCAM 模型加载 - 使用 eval.py 相同方式】")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"类别数: {num_classes}")
    print(f"设备: {device}")
    print(f"使用增强模型: {use_enhanced}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 【关键修复】使用与 eval.py 完全相同的方式创建模型
    # eval.py 使用 finetuning_exp.Exp 类，它会创建 Model 包装器
    print("\n[1/3] 创建模型结构 (使用 Exp 类，与 eval.py 相同)...")
    
    from projects.eval_tools.finetuning_exp import Exp
    
    # 创建 Exp 实例
    exp = Exp(batch_size=1)  # batch_size 不影响模型结构
    exp.num_classes = num_classes
    
    # 设置模型架构
    # 注意: finetuning_mae_exp.py 的 get_model() 会根据 use_enhanced_model 自动添加 _Enhanced 后缀
    # 所以这里只需要设置基础架构名称
    exp.encoder_arch = "HiFuse_Small"  # 基础架构名称
    
    if use_enhanced:
        exp.use_enhanced_model = True
        exp.enhancement_reduction = enhancement_reduction
        print(f"  使用增强版模型 (HiFuse_Small_Enhanced), reduction={enhancement_reduction}")
    else:
        exp.use_enhanced_model = False
        print(f"  使用基础模型 (HiFuse_Small)")
    
    # 禁用预训练（我们要加载自己的权重）
    exp.pretrained = False
    
    # 禁用 EMA（GradCAM 不需要）
    exp.model_ema = False
    
    # 获取 Model 包装器
    model_wrapper = exp.get_model()
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model_wrapper.parameters())
    trainable_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    print(f"  模型总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 【关键】加载权重到 Model 包装器（与 eval.py 相同）
    print("\n[2/3] 加载权重 (使用 set_model_weights，与 eval.py 相同)...")
    msg = set_model_weights(model_wrapper, model_path, weights_prefix="")
    
    # 详细打印加载结果
    print("\n[3/3] 权重加载结果:")
    print("-" * 40)
    
    # 缺失的keys（过滤掉 ema_model 的 keys）
    real_missing = [k for k in msg.missing_keys if not k.startswith('ema_model.')]
    if real_missing:
        print(f"  ⚠️  缺失的keys ({len(real_missing)}个):")
        # 按模块分组显示
        missing_groups = {}
        for key in real_missing:
            module = key.split('.')[0] if '.' in key else 'root'
            if module not in missing_groups:
                missing_groups[module] = []
            missing_groups[module].append(key)
        
        for module, keys in missing_groups.items():
            print(f"      [{module}]: {len(keys)}个")
            if len(keys) <= 3:
                for k in keys:
                    print(f"        - {k}")
    else:
        print("  ✅ 所有权重已成功加载!")
    
    # 未使用的keys
    if msg.unexpected_keys:
        print(f"\n  ⚠️  未使用的keys ({len(msg.unexpected_keys)}个):")
        for key in msg.unexpected_keys[:5]:
            print(f"      - {key}")
        if len(msg.unexpected_keys) > 5:
            print(f"      ... 还有 {len(msg.unexpected_keys) - 5} 个")
    else:
        print("  ✅ 没有多余的权重keys")
    
    # 检查关键模块是否加载
    print("\n  关键模块检查:")
    # 获取内部模型
    inner_model = model_wrapper.model
    key_modules = ['head', 'fu1', 'fu2', 'fu3', 'fu4', 'stages', 'layers']
    for module in key_modules:
        has_missing = any(module in k for k in real_missing)
        status = "⚠️ 部分缺失" if has_missing else "✅ 已加载"
        # 检查模型是否有这个模块
        has_module = any(module in name for name, _ in inner_model.named_modules())
        if has_module:
            print(f"      {module}: {status}")
    
    print("-" * 40)
    
    model_wrapper = model_wrapper.to(device)
    model_wrapper.eval()
    
    # 验证模型可以正常前向传播
    print("\n验证模型前向传播...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            # Model 包装器在 eval 模式下返回 (logits, head_loss)
            output = model_wrapper(dummy_input)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            print(f"  ✅ 前向传播成功! 输出shape: {logits.shape}")
            
            # 检查输出是否正常（不是全零或NaN）
            if torch.isnan(logits).any():
                print("  ⚠️ 警告: 输出包含NaN!")
            elif (logits == 0).all():
                print("  ⚠️ 警告: 输出全为0!")
            else:
                probs = torch.softmax(logits, dim=1)
                max_prob = probs.max().item()
                pred_class = probs.argmax(dim=1).item()
                print(f"  最大softmax概率: {max_prob:.4f} (预测类别: {pred_class})")
                if max_prob < 0.3:
                    print("  ⚠️ 警告: 最大概率较低，可能是随机输入导致")
                else:
                    print("  ✅ 模型输出正常!")
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("模型加载完成!")
    print("=" * 60 + "\n")
    
    # 【重要】返回内部模型用于 GradCAM
    # GradCAM 需要访问中间层 (fu1, fu2, fu3, fu4)，这些在 inner_model 中
    # 但我们需要保持 Model 包装器的 forward 行为，所以创建一个代理类
    class GradCAMModelProxy:
        """
        GradCAM 模型代理类
        - 使用 Model 包装器的 forward 方法（确保正确的推理行为）
        - 暴露内部模型的中间层（用于 GradCAM hook）
        """
        def __init__(self, model_wrapper):
            # 使用 object.__setattr__ 避免触发 __getattr__
            object.__setattr__(self, '_model_wrapper', model_wrapper)
            object.__setattr__(self, '_inner_model', model_wrapper.model)
        
        @property
        def model_wrapper(self):
            return object.__getattribute__(self, '_model_wrapper')
        
        @property
        def model(self):
            return object.__getattribute__(self, '_inner_model')
        
        def __call__(self, x, target=None):
            # 使用 Model 包装器的 forward
            output = self.model_wrapper(x, target)
            if isinstance(output, tuple):
                return output[0]  # 只返回 logits
            return output
        
        def eval(self):
            self.model_wrapper.eval()
            return self
        
        def train(self, mode=True):
            self.model_wrapper.train(mode)
            return self
        
        def to(self, device):
            self.model_wrapper.to(device)
            return self
        
        def zero_grad(self):
            self.model_wrapper.zero_grad()
        
        def parameters(self):
            return self.model_wrapper.parameters()
        
        def named_modules(self):
            return self.model.named_modules()
        
        def __getattr__(self, name):
            # 代理到内部模型的属性（用于访问 fu1, fu2 等）
            # 这个方法只在属性不存在时被调用
            return getattr(object.__getattribute__(self, '_inner_model'), name)
    
    return GradCAMModelProxy(model_wrapper)


def load_model_old(model_path, num_classes=7, device='cuda'):
    """旧版加载模型（备用）"""
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'projects', 'mae_lite'))
    
    from models_mae import main_model
    
    model = main_model(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 处理可能的key前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        # 【注意】这里跳过了head权重，会导致分类头随机初始化！
        if 'head.proto' in k or 'head.class_weights' in k:
            continue
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def process_image(image_path, transform=None):
    """
    处理输入图像
    
    【重要修复】归一化参数必须与训练时一致！
    训练时使用: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    之前错误使用了 ImageNet 默认值: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    这会导致模型输出随机结果（置信度约 0.24 = 1/8 类）
    """
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    
    if transform is None:
        # 【修复】使用与训练时相同的归一化参数
        # 参考: mae_lite/exps/timm_imagenet_exp.py 中的 data_transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    input_tensor = transform(img).unsqueeze(0)
    
    return input_tensor, original_img


def main():
    parser = argparse.ArgumentParser(description='多阶段Grad-CAM可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image_path', type=str, help='单张图像路径')
    parser.add_argument('--image_dir', type=str, help='图像目录（批量处理）')
    parser.add_argument('--output_dir', type=str, default='./cam_results', help='输出目录')
    parser.add_argument('--num_classes', type=int, default=7, help='类别数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--comparison', action='store_true', help='生成对比图')
    parser.add_argument('--row_labels', type=str, nargs='+', help='对比图行标签')
    parser.add_argument('--auto_select', action='store_true', help='自动选择最佳样本')
    parser.add_argument('--top_k', type=int, default=3, help='选择前k个最佳样本')
    parser.add_argument('--min_confidence', type=float, default=0.7, help='最小置信度阈值')
    parser.add_argument('--class_names', type=str, nargs='+', help='类别名称列表')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, args.num_classes, args.device)
    
    # 获取目标层
    target_layers = get_hifuse_target_layers(model)
    print(f"检测到的目标层: {list(target_layers.keys())}")
    
    if not target_layers:
        target_layers = get_hifuse_layers_v2(model)
        print(f"使用备用方法检测到的层: {list(target_layers.keys())}")
    
    if not target_layers:
        print("警告: 未能自动检测到目标层，请手动指定")
        return
    
    # 自动选择最佳样本模式
    if args.auto_select and args.image_dir:
        print(f"自动选择模式: 从 {args.image_dir} 中选择最佳样本...")
        best_samples = auto_select_and_visualize(
            model, 
            args.image_dir, 
            args.output_dir,
            samples_per_class=1,
            class_names=args.class_names,
            device=args.device
        )
        print(f"\n选择了 {len(best_samples)} 个最佳样本")
        for i, s in enumerate(best_samples):
            print(f"  {i+1}. {os.path.basename(s['path'])} - 综合得分: {s['combined_score']:.4f}")
        return
    
    # 创建GradCAM
    gradcam = GradCAM(model, target_layers)
    
    # 处理图像
    if args.image_path:
        # 单张图像
        input_tensor, original_img = process_image(args.image_path)
        input_tensor = input_tensor.to(args.device)
        
        cams, pred_class = gradcam(input_tensor)
        
        # 评估质量
        progressive_score, stage_scores = evaluate_progressive_quality(cams)
        print(f"\n热力图质量评估:")
        print(f"  渐进性得分: {progressive_score:.4f}")
        for stage, scores in stage_scores.items():
            print(f"  {stage}: 聚焦={scores['focus_score']:.3f}, 对比={scores['contrast_score']:.3f}")
        
        save_name = os.path.splitext(os.path.basename(args.image_path))[0]
        save_path = os.path.join(args.output_dir, f'{save_name}_gradcam.png')
        
        # 调整原图大小
        original_img_resized = cv2.resize(original_img, (224, 224))
        visualize_cam(original_img_resized, cams, save_path, f'Pred: Class {pred_class}')
        print(f"热力图已保存到: {save_path}")
    
    elif args.image_dir and args.comparison:
        # 批量处理并生成对比图
        image_files = sorted([f for f in os.listdir(args.image_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        images_data = []
        for img_file in image_files[:3]:  # 最多3行
            img_path = os.path.join(args.image_dir, img_file)
            input_tensor, original_img = process_image(img_path)
            input_tensor = input_tensor.to(args.device)
            
            cams, pred_class = gradcam(input_tensor)
            original_img_resized = cv2.resize(original_img, (224, 224))
            images_data.append((original_img_resized, cams))
        
        save_path = os.path.join(args.output_dir, 'gradcam_comparison.png')
        create_comparison_figure(images_data, save_path, args.row_labels)
    
    # 清理
    gradcam.remove_hooks()


# 命令行模式入口（已禁用，改用直接运行模式）
# if __name__ == '__main__':
#     main()


# ============== 热力图质量评估函数 ==============

def evaluate_cam_quality(cam, original_img=None):
    """
    评估单个CAM热力图的质量
    
    返回多个指标的字典:
    - focus_score: 聚焦度 (0-1)，越高表示热力图越集中
    - contrast_score: 对比度 (0-1)，越高表示高低激活区分越明显
    - coverage_score: 覆盖度 (0-1)，高激活区域占比
    - peak_intensity: 峰值强度
    - overall_score: 综合得分
    """
    cam = np.array(cam)
    if cam.max() > 0:
        cam_norm = cam / cam.max()
    else:
        cam_norm = cam
    
    # 1. 聚焦度：使用二阶矩（方差）来衡量分布的集中程度
    # 热力图越集中，方差越小，聚焦度越高
    h, w = cam_norm.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # 计算质心
    total_mass = cam_norm.sum() + 1e-8
    cx = (x_coords * cam_norm).sum() / total_mass
    cy = (y_coords * cam_norm).sum() / total_mass
    
    # 计算到质心的加权距离（二阶矩）
    dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
    variance = (dist_sq * cam_norm).sum() / total_mass
    max_variance = (h**2 + w**2) / 4  # 最大可能方差
    focus_score = 1 - np.sqrt(variance / max_variance)
    focus_score = np.clip(focus_score, 0, 1)
    
    # 2. 对比度：高激活区域和低激活区域的差异
    threshold = 0.5
    high_region = cam_norm[cam_norm > threshold]
    low_region = cam_norm[cam_norm <= threshold]
    
    if len(high_region) > 0 and len(low_region) > 0:
        contrast_score = high_region.mean() - low_region.mean()
    else:
        contrast_score = 0
    contrast_score = np.clip(contrast_score, 0, 1)
    
    # 3. 覆盖度：高激活区域占比（太大或太小都不好）
    coverage = (cam_norm > threshold).sum() / cam_norm.size
    # 理想覆盖度在10%-40%之间
    if coverage < 0.1:
        coverage_score = coverage / 0.1
    elif coverage > 0.4:
        coverage_score = max(0, 1 - (coverage - 0.4) / 0.6)
    else:
        coverage_score = 1.0
    
    # 4. 峰值强度
    peak_intensity = cam_norm.max()
    
    # 5. 综合得分（加权平均）
    overall_score = (
        0.35 * focus_score + 
        0.30 * contrast_score + 
        0.20 * coverage_score + 
        0.15 * peak_intensity
    )
    
    return {
        'focus_score': float(focus_score),
        'contrast_score': float(contrast_score),
        'coverage_score': float(coverage_score),
        'peak_intensity': float(peak_intensity),
        'overall_score': float(overall_score)
    }


def evaluate_progressive_quality(cams_dict):
    """
    评估多阶段CAM的渐进聚焦质量
    好的热力图应该从Stage1到Stage4逐渐聚焦
    
    Returns:
        progressive_score: 渐进性得分 (0-1)
        stage_scores: 各阶段的质量得分
    """
    stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    stage_scores = {}
    focus_values = []
    
    for stage in stage_names:
        if stage in cams_dict:
            scores = evaluate_cam_quality(cams_dict[stage])
            stage_scores[stage] = scores
            focus_values.append(scores['focus_score'])
    
    # 计算渐进性：后面的stage应该比前面更聚焦
    progressive_score = 0
    if len(focus_values) >= 2:
        # 检查是否单调递增
        increases = sum(1 for i in range(len(focus_values)-1) if focus_values[i+1] > focus_values[i])
        progressive_score = increases / (len(focus_values) - 1)
        
        # 额外奖励：最后一个stage的聚焦度
        if focus_values:
            progressive_score = 0.7 * progressive_score + 0.3 * focus_values[-1]
    
    return progressive_score, stage_scores


def select_best_samples(model, image_paths, device='cuda', top_k=3, min_confidence=0.7):
    """
    从多张图像中自动选择热力图效果最好的样本
    
    使用 pytorch-grad-cam 库，解决 inplace 操作导致的梯度问题
    
    Args:
        model: 模型
        image_paths: 图像路径列表
        device: 设备
        top_k: 选择前k个最好的
        min_confidence: 最小置信度阈值
        
    Returns:
        selected: 选中的图像信息列表，按质量排序
    """
    target_layers = get_hifuse_target_layers(model)
    if not target_layers:
        target_layers = get_hifuse_layers_v2(model)
    
    print(f"  目标层: {list(target_layers.keys())}")
    
    # 创建一个共享的 GradCAM 实例（pytorch-grad-cam 可以复用）
    gradcam = GradCAM(model, target_layers)
    
    results = []
    skipped_low_conf = 0
    processed = 0
    errors = 0
    
    for img_path in image_paths:
        try:
            input_tensor, original_img = process_image(img_path)
            input_tensor = input_tensor.to(device)
            
            # 获取预测和置信度
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                probs = F.softmax(output, dim=1)
                confidence, pred_class = probs.max(dim=1)
                confidence = confidence.item()
                pred_class = pred_class.item()
            
            processed += 1
            
            # 跳过低置信度样本
            if confidence < min_confidence:
                skipped_low_conf += 1
                continue
            
            # 生成CAM
            cams, _ = gradcam(input_tensor, target_class=pred_class)
            
            # 检查是否成功生成了CAM
            valid_cams = {k: v for k, v in cams.items() if v is not None and v.max() > 0}
            if not valid_cams:
                # 如果没有有效的CAM，跳过这张图
                continue
            
            # 评估质量
            progressive_score, stage_scores = evaluate_progressive_quality(cams)
            
            # 计算最终得分
            final_scores = [s['overall_score'] for s in stage_scores.values()]
            avg_quality = np.mean(final_scores) if final_scores else 0
            
            # 综合得分 = 质量 * 渐进性 * 置信度
            combined_score = avg_quality * 0.4 + progressive_score * 0.4 + confidence * 0.2
            
            results.append({
                'path': img_path,
                'pred_class': pred_class,
                'confidence': confidence,
                'progressive_score': progressive_score,
                'avg_quality': avg_quality,
                'combined_score': combined_score,
                'stage_scores': stage_scores,
                'cams': cams,
                'original_img': cv2.resize(original_img, (224, 224))
            })
            
        except Exception as e:
            errors += 1
            # 只打印前几个错误，避免刷屏
            if errors <= 3:
                print(f"处理 {img_path} 时出错: {e}")
            elif errors == 4:
                print(f"... 更多错误省略 ...")
            continue
    
    # 清理
    gradcam.remove_hooks()
    
    print(f"  处理统计: 总共处理 {processed} 张, 低置信度跳过 {skipped_low_conf} 张, 错误 {errors} 张, 有效 {len(results)} 张")
    
    # 按综合得分排序
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return results[:top_k]


def auto_select_and_visualize(model, image_dir, output_dir, 
                               samples_per_class=1, 
                               class_names=None,
                               device='cuda'):
    """
    自动从数据集中选择每个类别最好的样本并生成可视化
    
    Args:
        model: 模型
        image_dir: 图像目录（可以有子目录按类别组织）
        output_dir: 输出目录
        samples_per_class: 每个类别选择的样本数
        class_names: 类别名称列表
        device: 设备
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有图像
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, f))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 选择最佳样本
    print("正在评估图像质量...")
    best_samples = select_best_samples(model, image_paths, device, top_k=10)
    
    # 按类别分组
    class_samples = {}
    for sample in best_samples:
        cls = sample['pred_class']
        if cls not in class_samples:
            class_samples[cls] = []
        if len(class_samples[cls]) < samples_per_class:
            class_samples[cls].append(sample)
    
    # 生成可视化
    images_data = []
    row_labels = []
    
    for cls in sorted(class_samples.keys()):
        for sample in class_samples[cls]:
            images_data.append((sample['original_img'], sample['cams']))
            label = class_names[cls] if class_names and cls < len(class_names) else f'Class {cls}'
            row_labels.append(label)
    
    if images_data:
        # 生成对比图
        comparison_path = os.path.join(output_dir, 'best_gradcam_comparison.png')
        create_comparison_figure(images_data, comparison_path, row_labels)
        
        # 保存质量报告
        report_path = os.path.join(output_dir, 'quality_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Grad-CAM 质量评估报告\n")
            f.write("=" * 50 + "\n\n")
            for i, sample in enumerate(best_samples):
                f.write(f"样本 {i+1}: {os.path.basename(sample['path'])}\n")
                f.write(f"  预测类别: {sample['pred_class']}\n")
                f.write(f"  置信度: {sample['confidence']:.4f}\n")
                f.write(f"  渐进性得分: {sample['progressive_score']:.4f}\n")
                f.write(f"  平均质量: {sample['avg_quality']:.4f}\n")
                f.write(f"  综合得分: {sample['combined_score']:.4f}\n")
                f.write(f"  各阶段得分:\n")
                for stage, scores in sample['stage_scores'].items():
                    f.write(f"    {stage}: 聚焦={scores['focus_score']:.3f}, "
                           f"对比={scores['contrast_score']:.3f}, "
                           f"覆盖={scores['coverage_score']:.3f}\n")
                f.write("\n")
        
        print(f"质量报告已保存到: {report_path}")
    
    return best_samples


# ============== 简化版本：直接在脚本中使用 ==============

def generate_gradcam_simple(model, image_path, output_path, device='cuda'):
    """
    简化版Grad-CAM生成函数
    
    Args:
        model: 已加载的模型
        image_path: 输入图像路径
        output_path: 输出路径
        device: 设备
    """
    # 获取目标层
    target_layers = get_hifuse_target_layers(model)
    
    # 创建GradCAM
    gradcam = GradCAM(model, target_layers)
    
    # 处理图像
    input_tensor, original_img = process_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # 生成CAM
    cams, pred_class = gradcam(input_tensor)
    
    # 可视化
    original_img_resized = cv2.resize(original_img, (224, 224))
    visualize_cam(original_img_resized, cams, output_path, f'Predicted: Class {pred_class}')
    
    gradcam.remove_hooks()
    return cams, pred_class


def get_hifuse_layers_v2(model):
    """
    针对HiFuse-Small的层级获取（更精确的版本）
    """
    target_layers = {}
    
    # 遍历模型找到合适的层
    def find_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 查找包含 'block' 或 'stage' 或 'layer' 的层
            if any(key in name.lower() for key in ['block', 'stage', 'layer', 'conv']):
                # 检查是否是Sequential或有forward方法
                if hasattr(child, 'forward'):
                    target_layers[full_name] = child
            
            # 递归查找
            find_layers(child, full_name)
    
    find_layers(model)
    
    # 选择4个代表性的层
    layer_names = list(target_layers.keys())
    if len(layer_names) >= 4:
        # 均匀选择4个层
        indices = np.linspace(0, len(layer_names) - 1, 4, dtype=int)
        selected = {f'Stage {i+1}': target_layers[layer_names[idx]] for i, idx in enumerate(indices)}
        return selected
    
    return target_layers


def batch_generate_gradcam(model, image_paths, output_dir, row_labels=None, device='cuda'):
    """
    批量生成Grad-CAM并创建对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    target_layers = get_hifuse_target_layers(model)
    if not target_layers:
        target_layers = get_hifuse_layers_v2(model)
    
    gradcam = GradCAM(model, target_layers)
    
    images_data = []
    for img_path in image_paths:
        input_tensor, original_img = process_image(img_path)
        input_tensor = input_tensor.to(device)
        
        cams, pred_class = gradcam(input_tensor)
        original_img_resized = cv2.resize(original_img, (224, 224))
        images_data.append((original_img_resized, cams))
        
        # 同时保存单张图
        save_name = os.path.splitext(os.path.basename(img_path))[0]
        single_path = os.path.join(output_dir, f'{save_name}_cam.png')
        visualize_cam(original_img_resized, cams, single_path)
    
    # 生成对比图
    comparison_path = os.path.join(output_dir, 'gradcam_comparison.png')
    create_comparison_figure(images_data, comparison_path, row_labels)
    
    gradcam.remove_hooks()
    return images_data


# ============== 直接运行配置 ==============
# 修改下面的路径后直接运行即可

if __name__ == '__main__':
    import json
    import random
    
    # ========== 【选择数据加载模式】 ==========
    # "public": 公共数据集，直接使用 VAL_DIR 路径
    # "private": 私有数据集，使用 read_split_data 划分方法
    DATA_MODE = "private"  # 改成 "private" 使用私有数据集划分
    
    # ========== 私有数据集的划分函数（保留备用） ==========
    def read_split_data(root1: str, root2: str, root3: str):
        """
        私有数据集划分函数
        root1: test目录
        root2: train目录 (super_no_aug)
        root3: val目录
        """
        random.seed(1)
        assert os.path.exists(root1), f"dataset root: {root1} does not exist."

        flower_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
        flower_class.sort()
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        
        train_images_path = []
        train_images_label = []
        val_images_path = []
        val_images_label = []
        supported = [".jpg", ".JPG", ".png", ".PNG"]
        
        for cla in flower_class:
            cla_path = os.path.join(root1, cla)  # test
            cla_path1 = os.path.join(root2, cla)  # train
            cla_path2 = os.path.join(root3, cla)  # val
            
            image = [i[:-4] for i in os.listdir(cla_path)]
            image1 = [i[:-4] for i in os.listdir(cla_path2)]
            image_class = class_indices[cla]
            
            image.extend(image1)
            val_path = [os.path.join(root2, cla, i) for i in os.listdir(cla_path1) if any(img in i for img in image)]
            train_path = [os.path.join(root2, cla, i) for i in os.listdir(cla_path1) if all(img not in i for img in image)]

            for i in val_path:
                val_images_path.append(i)
                val_images_label.append(image_class)
            for i in train_path:
                train_images_path.append(i)
                train_images_label.append(image_class)

        print(f"{len(train_images_path)} images for training.")
        print(f"{len(val_images_path)} images for validation.")
        return train_images_path, train_images_label, val_images_path, val_images_label
    
    # ========== 公共数据集加载函数 ==========
    def load_public_dataset(val_dir):
        """
        公共数据集加载函数
        直接从 val_dir 读取所有图片
        目录结构: val_dir/class_name/image.jpg
        """
        val_images_path = []
        val_images_label = []
        
        assert os.path.exists(val_dir), f"val_dir: {val_dir} does not exist."
        
        classes = [cla for cla in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, cla))]
        classes.sort()
        class_indices = {k: v for v, k in enumerate(classes)}
        
        for cla in classes:
            cla_path = os.path.join(val_dir, cla)
            for img_name in os.listdir(cla_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    val_images_path.append(os.path.join(cla_path, img_name))
                    val_images_label.append(class_indices[cla])
        
        print(f"Loaded {len(val_images_path)} images from {val_dir}")
        print(f"Classes: {classes}")
        return val_images_path, val_images_label, classes
    
    # ========== 配置区域 ==========
    
    # 模型权重路径
    MODEL_PATH = "/home/backup/lh/KD/outputs/HiFuse_Small_1e-5-0.05/ft_progressive_v3_eval/90last_epoch_best_ckpt.checkpoints.tar"
    
    # 是否使用增强版模型 (多原型分类头 + MHF增强模块)
    USE_ENHANCED_MODEL = True  # 改成 True 使用增强版模型
    ENHANCEMENT_REDUCTION = 0.0625  # 增强模块的reduction参数
    
    # === 公共数据集配置 ===
    # 直接指定 val 目录路径
    VAL_DIR = "/home/backup/lh/KD/data/kvasir/val/"
    
    # === 私有数据集配置（DATA_MODE="private" 时使用） ===
    TEST_DIR = "/home/backup/lh/lung/data/kd3/test_20260127"
    TRAIN_DIR = "/home/backup/lh/lung/data/super_no_aug_20260127"
    PRIVATE_VAL_DIR = "/home/backup/lh/lung/data/kd3/val_20260127"
    
    # 输出目录
    OUTPUT_DIR = "./cam_results12"
    
    # 类别数
    NUM_CLASSES = 3  # 公共数据集的类别数，私有数据集改成 3
    
    # 类别名称（Kvasir8 数据集）
    # 按字母顺序排列的类别
    # CLASS_NAMES = [
    #     'dyed-lifted-polyps',      # 0 - 染色隆起息肉
    #     'dyed-resection-margins',  # 1 - 染色切除边缘
    #     'esophagitis',             # 2 - 食管炎
    #     'normal-cecum',            # 3 - 正常盲肠
    #     'normal-pylorus',          # 4 - 正常幽门
    #     'normal-z-line',           # 5 - 正常Z线
    #     'polyps',                  # 6 - 息肉
    #     'ulcerative-colitis'       # 7 - 溃疡性结肠炎
    # ]
    # 私有数据集（肺结节）
    CLASS_NAMES = ['AISMIA', 'G1G2', 'G3']
    
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 选择最好的几张图
    TOP_K = 3
    
    # ================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Grad-CAM 热力图生成工具")
    print(f"数据加载模式: {DATA_MODE}")
    print("=" * 50)
    
    # 根据模式加载数据
    print("\n正在加载数据集...")
    if DATA_MODE == "public":
        val_images_path, val_images_label, detected_classes = load_public_dataset(VAL_DIR)
        # 如果没有手动设置类别名，使用检测到的类别名
        if CLASS_NAMES == ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']:
            CLASS_NAMES = detected_classes
    else:  # private
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
            TEST_DIR, TRAIN_DIR, PRIVATE_VAL_DIR
        )
    
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")
    print(f"验证集图片数: {len(val_images_path)}")
    print()
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE, 
                       use_enhanced=USE_ENHANCED_MODEL, 
                       enhancement_reduction=ENHANCEMENT_REDUCTION)
    print("模型加载完成!")
    
    # 获取目标层
    target_layers = get_hifuse_target_layers(model)
    if not target_layers:
        target_layers = get_hifuse_layers_v2(model)
    print(f"检测到的目标层: {list(target_layers.keys())}")
    
    # 从验证集中选择最佳样本
    # 降低置信度阈值到0.1，确保能选到样本
    MIN_CONFIDENCE = 0.1
    print(f"\n从验证集中选择最佳样本 (共 {len(val_images_path)} 张, 最小置信度: {MIN_CONFIDENCE})...")
    # 先选更多样本，然后从中挑选不同类别的
    all_samples = select_best_samples(model, val_images_path, DEVICE, top_k=200, min_confidence=MIN_CONFIDENCE)
    
    # 策略：确保每个类别都有样本
    best_by_class = {}
    for s in all_samples:
        cls = s['pred_class']
        if cls not in best_by_class:
            best_by_class[cls] = s
    
    # 按类别顺序添加（确保8个类别都有）
    best_samples = [best_by_class[cls] for cls in sorted(best_by_class.keys())]
    
    # 如果不够 TOP_K 个，从剩余样本中补充
    if len(best_samples) < TOP_K:
        used_paths = {s['path'] for s in best_samples}
        for s in all_samples:
            if s['path'] not in used_paths:
                best_samples.append(s)
                used_paths.add(s['path'])
                if len(best_samples) >= TOP_K:
                    break
    
    # 最终取前 TOP_K 个
    best_samples = best_samples[:TOP_K]
    
    if best_samples:
        print(f"\n选择了 {len(best_samples)} 个最佳样本:")
        for i, s in enumerate(best_samples):
            cls_name = CLASS_NAMES[s['pred_class']] if s['pred_class'] < len(CLASS_NAMES) else f"Class {s['pred_class']}"
            print(f"  {i+1}. {os.path.basename(s['path'])} - 类别: {cls_name}, 置信度: {s['confidence']:.4f}, 综合得分: {s['combined_score']:.4f}")
        
        # 生成可视化
        images_data = [(s['original_img'], s['cams']) for s in best_samples]
        row_labels = [CLASS_NAMES[s['pred_class']] if s['pred_class'] < len(CLASS_NAMES) else f"Class {s['pred_class']}" for s in best_samples]
        
        comparison_path = os.path.join(OUTPUT_DIR, 'best_gradcam_comparison.png')
        create_comparison_figure(images_data, comparison_path, row_labels)
        
        # 同时保存单张图
        for s in best_samples:
            cls_name = CLASS_NAMES[s['pred_class']] if s['pred_class'] < len(CLASS_NAMES) else f"Class {s['pred_class']}"
            save_name = os.path.splitext(os.path.basename(s['path']))[0]
            single_path = os.path.join(OUTPUT_DIR, f'{save_name}_cam.png')
            visualize_cam(s['original_img'], s['cams'], single_path, cls_name)
        
        print(f"\n结果已保存到: {OUTPUT_DIR}")
    else:
        print("未找到合适的样本!")
