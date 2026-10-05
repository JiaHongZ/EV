"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, \
    get_act_layer, get_norm_layer, create_classifier
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs, register_model_deprecations
import torch
import numpy as np
import torch.nn.functional as F
from timm.models.resnet import ResNet
from timm.models.fastvit import convolutional_stem

import timm.models.retinal.retina_module as RM
import timm.models.retinal.scanpath as SP

import timm

# for convolutional network and fastvit network
__all__ = ['EVC', 'EVT_FastViT']  # model_registry will add each entrypoint fn to this
   
class EVC(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            retina_block=None,
            log_r=True,
            use_retina=False,
            use_embodied=False,
            pre_trained=False,
            use_residual=False,
            use_retina_field=False,
            rf2=False,
            retina_fixed: bool = True,
            retina_sampling_model: str = 'rl',
            retina_size: int = 224,
            patch_number=2,  # 眼动点数量
            retina_field=1,
            model_name='evc_resnet18', 
            pretrained=True, checkpoint_path='',
            rl_hidden_size = 128,
            rl_std=0.05,
            scanpath_size = 5,
            zero_init_last=True,  **kwargs,
    ):

        super(EVC, self).__init__()
        self.num_classes = num_classes
        
        self.use_retina = use_retina
        self.retina_fixed = retina_fixed
        self.num_patches = patch_number
        self.retina_size = retina_size
        self.use_embodied = use_embodied
        self.pre_trained = pre_trained
        self.rf2 = rf2
        self.in_chans = in_chans
        self.patch_number = patch_number-1 
        self.retina_sampling_model = retina_sampling_model
        self.w_scale = 2
        self.retina = RM.Retina_SR_iter_residual(
            image_H=retina_size,
            image_W=retina_size,
            retinal_H=retina_size,
            retinal_W=retina_size,
            w_scale=self.w_scale,
        )
        if use_embodied:
            self.pos_embedding = nn.Sequential(
                    nn.Linear(2, 1),  # 2维坐标转为embedding
                    nn.Sigmoid(),
                )
            self.sr_embedding = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid(),
            )
            self.final_embed = nn.Sequential(
                nn.Conv2d((3*2 + 2), 3, 5, 1, 2), 
                nn.GELU(),
            )
        self.retina_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if '18' in model_name:
            checkpoint_path = '/zjh/data/resnet18-f37072fd.pth'
            self.model = timm.create_model(
                'resnet18.tv_in1k', pretrained=True, pretrained_cfg_overlay=dict(file=checkpoint_path), features_only=False, num_classes=num_classes, in_chans=in_chans)
            self.hidden_size = self.model.channels[-1]
            self.backbone_feature_dim = self.model.num_features if hasattr(self.model, 'num_features') else 512
            
        if '50' in model_name:
            checkpoint_path = '/zjh/data/resnet50-0676ba61.pth'
            self.model = timm.create_model(
                'resnet50.tv_in1k', pretrained=True, pretrained_cfg_overlay=dict(file=checkpoint_path), features_only=False, num_classes=num_classes, in_chans=in_chans)
            self.hidden_size = 2048
            self.backbone_feature_dim = self.model.num_features if hasattr(self.model, 'num_features') else 2048
            
        if retina_sampling_model == 'rl':
            self.movements = patch_number
            self.rnn_hidden_size = 64 
            self.rnn = SP.LSTMCoreNetwork(64 * 1 * 1, self.rnn_hidden_size, 2)
            self.locator = SP.LocationNetwork(self.rnn_hidden_size, 2, 0.2) 
            self.sr_net = SP.SRNetwork(self.rnn_hidden_size, 1, self.w_scale, 0.05)
            self.baseliner = SP.BaselineNetwork(self.rnn_hidden_size, 1)
            self.projection_rank = 64 
            self.h_projection_A = nn.Linear(self.rnn_hidden_size, self.projection_rank, bias=False)
            self.h_projection_B = nn.Linear(self.projection_rank, self.backbone_feature_dim, bias=False)
            self.c_t = None
            
    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def _embed(self, x, l_t, rf, s_t):
        b,c,h,w = x.shape
        pos_embed = self.pos_embedding(l_t).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        sr_embed = self.sr_embedding(s_t).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        x = torch.cat([x, pos_embed, sr_embed],1)
        x = self.final_embed(x)
        return x
    def forward(self, x, l_t=torch.zeros(1,2).cuda(), h_t=torch.zeros(1,2048).cuda(), test=True):
        if l_t is None:
            l_t = torch.zeros(x.size()[0], 2).to(x.device)
        
        # Get the correct rnn_hidden_size
        rnn_hidden_size = getattr(self, 'rnn_hidden_size', 256)
        
        if h_t is None:
            # Use rnn_hidden_size instead of large hidden_size
            h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        else:
            # Check if h_t has the correct dimension, if not, reinitialize it
            if h_t.shape[1] != rnn_hidden_size:
                import warnings
                warnings.warn(f"h_t dimension mismatch: expected {rnn_hidden_size}, got {h_t.shape[1]}. Reinitializing.")
                h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        s_t = torch.ones(x.size()[0],1).cuda() / self.w_scale
        log_pils = []
        log_pils_srs = []
        
        baselines = []
        xs = []
        l_ts = []
        l_srs = []
        
        x_ = x
        l_ts.append(l_t.unsqueeze(1))
        c_t = torch.zeros_like(h_t).to(h_t.device)
        h_ts = []
        h_ts_projected = []  # Store projected hidden states for backbone fusion
        
        for i in range(self.movements):
            x_i, l_t, rf = self.retina(x_, l_t, s_t)
            i_t = self._embed(x_i, l_t, rf, s_t)

            x = self.model.conv1(i_t)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
            
            x_rnn = self.retina_avgpool(x).flatten(1)
            out, h_t, c_t = self.rnn(x_rnn, (h_t, c_t))
            log_pil, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t).squeeze(-1)
            log_pil_sr, s_t = self.sr_net(h_t)
            
            xs.append(x.unsqueeze(1))
            baselines.append(b_t)
            log_pils.append(log_pil)
            log_pils_srs.append(log_pil_sr)
            l_ts.append(l_t.unsqueeze(1))
            l_srs.append(s_t.unsqueeze(1))
            h_ts.append(h_t.clone())
            h_t_projected = self.h_projection_B(self.h_projection_A(h_t))
            h_ts_projected.append(h_t_projected)
        # 返回每个时间步的预测，用于AdaptiveNN风格的confidence递增奖励
        if self.training:
            pres = []
            x = torch.cat(xs, 1).mean(1)
            x1 = self.model.layer1(x)
            x2 = self.model.layer2(x1)
            x3 = self.model.layer3(x2)
            x4 = self.model.layer4(x3)
            x = self.model.global_pool(x4)
            x = x.view(x.shape[0], -1)
            # 使用对应时间步的hidden state
            y = self.model.fc(x + h_ts_projected[-1])
            pres.append(y)
            y_final = pres[-1]
            return y_final, l_ts, baselines, log_pils, l_srs, log_pils_srs, pres
        else:
            # 可解释性分析
            # pres = []
            # for i in range(self.movements):
            #     # 使用到当前时间步为止的所有特征
            #     x = torch.cat(xs[:i+1], 1).mean(1)
            #     x1 = self.model.layer1(x)
            #     x2 = self.model.layer2(x1)
            #     x3 = self.model.layer3(x2)
            #     x4 = self.model.layer4(x3)
            #     x = self.model.global_pool(x4)
            #     x = x.view(x.shape[0], -1)
            #     # 使用当前时间步的投影后的hidden state与backbone特征融合
            #     # 每个时间步使用对应的hidden state，这样每个movement的预测会不同
            #     y = self.model.fc(x + h_ts_projected[i])
            #     pres.append(y)
            
            # 直接推理节约时间   
            pres = []
            # 使用到当前时间步为止的所有特征
            x = torch.cat(xs, 1).mean(1)
            x1 = self.model.layer1(x)
            x2 = self.model.layer2(x1)
            x3 = self.model.layer3(x2)
            x4 = self.model.layer4(x3)
            x = self.model.global_pool(x4)
            x = x.view(x.shape[0], -1)
            y = self.model.fc(x + h_ts_projected[-1])
            
            pres.append(y)
            
            y_final = pres[-1]

            return y, l_ts, baselines, log_pils, l_srs, log_pils_srs, pres
  
    def confidence(self, x, l_t=torch.zeros(1,2).cuda(), h_t=torch.zeros(1,64).cuda(), em=None, test=True):
        if em is None:
            em = self.movements

        if l_t is None:
            l_t = torch.zeros(x.size()[0], 2).to(x.device)
        
        rnn_hidden_size = getattr(self, 'rnn_hidden_size', 256)
        
        if h_t is None:
            h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        else:
            if h_t.shape[1] != rnn_hidden_size:
                import warnings
                warnings.warn(f"h_t dimension mismatch: expected {rnn_hidden_size}, got {h_t.shape[1]}. Reinitializing.")
                h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        s_t = torch.ones(x.size()[0],1).cuda() / self.w_scale
        log_pils = []
        log_pils_srs = []
        
        baselines = []
        xs = []
        l_ts = []
        l_srs = []
        
        x_ = x
        l_ts.append(l_t.unsqueeze(1))
        c_t = torch.zeros_like(h_t).to(h_t.device)
        h_ts = []
        h_ts_projected = []  # Store projected hidden states for backbone fusion
        
        for i in range(em):
            x_i, l_t, rf = self.retina(x_, l_t, s_t)
            i_t = self._embed(x_i, l_t, rf, s_t)

            x = self.model.conv1(i_t)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)

            x_rnn = self.retina_avgpool(x).flatten(1)
            out, h_t, c_t = self.rnn(x_rnn, (h_t, c_t))
            log_pil, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t).squeeze(-1)
            log_pil_sr, s_t = self.sr_net(h_t)
            
            xs.append(x.unsqueeze(1))
            baselines.append(b_t)
            log_pils.append(log_pil)
            log_pils_srs.append(log_pil_sr)
            l_ts.append(l_t.unsqueeze(1))
            l_srs.append(s_t.unsqueeze(1))
            h_ts.append(h_t.clone())
            h_t_projected = self.h_projection_B(self.h_projection_A(h_t))
            h_ts_projected.append(h_t_projected)
        else:
            # 可解释性分析
            pres = []
            for i in range(em):
                # 使用到当前时间步为止的所有特征
                x = torch.cat(xs[:i+1], 1).mean(1)
                x1 = self.model.layer1(x)
                x2 = self.model.layer2(x1)
                x3 = self.model.layer3(x2)
                x4 = self.model.layer4(x3)
                x = self.model.global_pool(x4)
                x = x.view(x.shape[0], -1)
                y = self.model.fc(x + h_ts_projected[i])
                pres.append(y)
            y_final = pres[-1]

            return y, l_ts, baselines, log_pils, l_srs, log_pils_srs, pres

class EVT_FastViT(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            in_chans=3,
            retina_block=None,
            log_r=True,
            use_retina=False,
            use_embodied=False,
            pre_trained=False,
            use_residual=False,
            use_retina_field=False,
            rf2=False,
            retina_fixed: bool = True,
            retina_sampling_model: str = 'rl',
            retina_size: int = 224,
            patch_number=16,  # 眼动点数量
            retina_field=1,
            model_name='evt_fastvit_sa12', 
            pretrained=True, checkpoint_path='',
            rl_hidden_size = 128,
            rl_std=0.05,
            scanpath_size = 5,
            zero_init_last=True,  **kwargs,
    ):

        super(EVT_FastViT, self).__init__()
        self.num_classes = num_classes
        
        self.use_retina = use_retina
        self.retina_fixed = retina_fixed
        self.num_patches = patch_number
        self.retina_size = retina_size
        self.use_embodied = use_embodied
        self.pre_trained = pre_trained
        self.rf2 = rf2
        self.in_chans = in_chans
        self.patch_number = patch_number-1 #overview其实算法中心眼动点，所以实际应该减一个
        self.retina_sampling_model = retina_sampling_model
        self.w_scale = 2
        # retina and eyemovement
        self.retina = RM.Retina_SR_iter_residual(
            image_H=retina_size,  # 输出图像大小(retina size)
            image_W=retina_size,
            retinal_H=retina_size,  # 中间变换图像大小
            retinal_W=retina_size,
            sampling_model='gaussian',  # 眼动点选取模式
            w_scale=self.w_scale,
        )
        if use_embodied:
            self.pos_embedding = nn.Sequential(
                    nn.Linear(2, 1),  # 2维坐标转为embedding
                    nn.Sigmoid(),
                )
            self.sr_embedding = nn.Sequential(
                nn.Linear(1, 1),  # 2维坐标转为embedding
                nn.Sigmoid(),
            )
            self.final_embed = nn.Sequential(
                nn.Conv2d((3 * 2 + 2), 3, 5, 1, 2),
                nn.GELU(),
            )
        self.retina_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if model_name[4:] == 'fastvit_sa12':
            checkpoint_path = './pretrained/fastvit_sa12.bin'
        if model_name[4:] == 'fastvit_sa24':
            checkpoint_path = './pretrained/fastvit_sa24.bin'
        self.model = timm.create_model(
            model_name[4:], pretrained=pretrained, pretrained_cfg_overlay=dict(file=checkpoint_path), features_only=False, num_classes=num_classes, in_chans=in_chans, **kwargs
        )
        self.hidden_size = 512
        if retina_sampling_model == 'rl':
            self.movements = patch_number
            self.rnn_hidden_size = 64  # Fixed small hidden size for RNN
            self.rnn = SP.LSTMCoreNetwork(64 * 1 * 1, self.rnn_hidden_size, 2)
            self.sr_net = SP.SRNetwork(self.rnn_hidden_size, 1, self.w_scale, 0.05)
            self.baseliner = SP.BaselineNetwork(self.rnn_hidden_size, 1)
            self.h_projection_A = nn.Linear(self.rnn_hidden_size, self.projection_rank, bias=False)
            self.h_projection_B = nn.Linear(self.projection_rank, 512, bias=False)
            self.c_t = None
            
    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
    def _embed(self, x, l_t, rf, s_t):
        b,c,h,w = x.shape
        pos_embed = self.pos_embedding(l_t).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        sr_embed = self.sr_embedding(s_t).unsqueeze(2).unsqueeze(3).repeat(1,1,h,w)
        x = torch.cat([x, pos_embed, sr_embed],1)
        x = self.final_embed(x)
        return x
    
    def forward(self, x, l_t=torch.zeros(1,2).cuda(), h_t=torch.zeros(1,64).cuda(), test=True):
        if l_t is None:
            l_t = torch.zeros(x.size()[0], 2).to(x.device)
        
        rnn_hidden_size = getattr(self, 'rnn_hidden_size', 256)
        
        if h_t is None:
            h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        else:
            if h_t.shape[1] != rnn_hidden_size:
                import warnings
                warnings.warn(f"h_t dimension mismatch: expected {rnn_hidden_size}, got {h_t.shape[1]}. Reinitializing.")
                h_t = torch.zeros(x.size()[0], rnn_hidden_size).to(x.device)
        s_t = torch.ones(x.size()[0],1).cuda() / self.w_scale
        log_pils = []
        log_pils_srs = []
        
        baselines = []
        xs = []
        l_ts = []
        l_srs = []
        
        x_ = x
        l_ts.append(l_t.unsqueeze(1))
        c_t = torch.zeros_like(h_t).to(h_t.device)
        h_ts = []
        h_ts_projected = []  # Store projected hidden states for backbone fusion
        
        for i in range(self.movements):
            x_i, l_t, rf = self.retina(x_, l_t, s_t)
            i_t = self._embed(x_i, l_t, rf, s_t)
            
            x = self.model.stem(i_t)
            
            x_rnn = self.retina_avgpool(x).flatten(1)
            out, h_t, c_t = self.rnn(x_rnn, (h_t, c_t))
            log_pil, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t).squeeze(-1)
            log_pil_sr, s_t = self.sr_net(h_t)
            
            xs.append(x.unsqueeze(1))
            baselines.append(b_t)
            log_pils.append(log_pil)
            log_pils_srs.append(log_pil_sr)
            l_ts.append(l_t.unsqueeze(1))
            l_srs.append(s_t.unsqueeze(1))
            h_ts.append(h_t.clone())
            h_t_projected = self.h_projection_B(self.h_projection_A(h_t))
            h_ts_projected.append(h_t_projected)
            
        if self.training:
            pres = []
            x = torch.cat(xs, 1).mean(1)
            outs = []
            for idx, block in enumerate(self.model.stages):
                x = block(x)
                if self.model.fork_feat:
                    if idx in self.model.out_indices:
                        norm_layer = getattr(self, f"norm{idx}")
                        x_out = norm_layer(x)
                        outs.append(x_out)
            if self.model.fork_feat:
                return outs
        
            x = self.model.final_conv(x + h_ts_projected[-1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,x.shape[-2],x.shape[-1]))
            # 循环
            y = self.model.head(x)
            
            pres.append(y)
            y_final = pres[-1]
            return y_final, l_ts, baselines, log_pils, l_srs, log_pils_srs, pres
        else:
            pres = []
            for i in range(self.movements):
                x = torch.cat(xs[:i+1], 1).mean(1)
                outs = []
                for idx, block in enumerate(self.model.stages):
                    x = block(x)
                    if self.model.fork_feat:
                        if idx in self.model.out_indices:
                            norm_layer = getattr(self, f"norm{idx}")
                            x_out = norm_layer(x)
                            outs.append(x_out)
                if self.model.fork_feat:
                    return outs
            
                x = self.model.final_conv(x + h_ts_projected[-1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,x.shape[-2],x.shape[-1]))
                # 循环
                y = self.model.head(x)
                
                pres.append(y)
            y_final = pres[-1]
            return y_final, l_ts, baselines, log_pils, l_srs, log_pils_srs, pres
        
def _create_resnet_retina(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(EVC, variant, pretrained, **kwargs)
def _create_rdtfastvit_retina(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(EVT_FastViT, variant, pretrained, **kwargs)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def _tcfg(url='', **kwargs):
    return _cfg(url=url, **dict({'interpolation': 'bicubic'}, **kwargs))


def _ttcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'test_input_size': (3, 288, 288), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models',
    }, **kwargs))


def _rcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'crop_pct': 0.95, 'test_input_size': (3, 288, 288), 'test_crop_pct': 1.0,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476'
    }, **kwargs))


def _r3cfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'input_size': (3, 160, 160), 'pool_size': (5, 5),
        'crop_pct': 0.95, 'test_input_size': (3, 224, 224), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476',
    }, **kwargs))


def _gcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic',
        'origin_url': 'https://cv.gluon.ai/model_zoo/classification.html',
    }, **kwargs))

@register_model
def evc_resnet18(pretrained=False, **kwargs) -> EVC:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(
        model_name='evc_resnet18',
        use_retina=True,
        use_embodied=True,
        retina_block=None,
        retina_field=1, retina_fixed=True, **kwargs
    )
    return _create_resnet_retina('evc_resnet18', pretrained, **dict(model_args, **kwargs))

@register_model
def rdc_resnet50(pretrained=False, **kwargs) -> EVC:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(
        model_name='rdc_resnet50',
        use_retina=True,
        use_embodied=True,
        retina_block=None,
        retina_field=1, retina_fixed=True,
        **kwargs
    )
    return _create_resnet_retina('rdc_resnet50', pretrained, **dict(model_args, **kwargs))

@register_model
def evt_fastvit_sa12(pretrained=False, **kwargs) -> EVT_FastViT:
    model_args = dict(
        model_name='evt_fastvit_sa12',
        use_retina=True,
        use_embodied=True,
        retina_block=None,
        retina_field=1, retina_fixed=True,
        **kwargs
    )
    return _create_rdtfastvit_retina('evt_fastvit_sa12', pretrained, **dict(model_args, **kwargs))
