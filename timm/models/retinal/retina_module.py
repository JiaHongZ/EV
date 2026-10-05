import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import itertools
import numpy as np
import math
import random
import os
# 固定随机数种子
def setup_seed(seed):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法
setup_seed(42)

class retina_polar(nn.Module):
    '''
    Log polar transformation
    '''

    def __init__(
            self,
            r_min=0.05,
            r_max=0.8,
            H=5,
            W=12,
            log_r=True,
            retina_field=1,
    ):
        super(retina_polar, self).__init__()
        if log_r:
            dr = (np.log(r_max * H) - np.log(r_min * H)) / H
        else:
            dr = ((r_max * H) - (r_min * H)) / H
        grid_2d = torch.empty(
            [H, W, 2]
        )
        angles = torch.empty(
            [H, W, 1]
        )
        for h in range(H):
            # radius = sample_r[h]
            radius = r_min*h/H*np.exp(dr*h) # 这里，如果不要 h/H，就没有中间放大的效果了
            for w in range(W):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.H = H
        self.W = W
        self.retina_field = retina_field
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)

    def get_grid(self, b):
        radius = self.radius[None].clone().repeat(b, 1, 1, 1)
        angles = self.angles[None].clone().repeat(b, 1, 1, 1)
        angle = angles

        grid = torch.zeros_like(radius).to(device=radius.device)
        grid[:, :, :, 0] = radius[:, :, :, 0] * torch.sin(angle[:, :, :, 0])
        grid[:, :, :, 1] = radius[:, :, :, 1] * torch.cos(angle[:, :, :, 0])
        return grid
    def forward(self, x, l_t_prev):
        batch_size, *_ = x.shape
        l_t_prev = l_t_prev.to(x.device)
        grid_2d_batch = self.get_grid(batch_size) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border', align_corners=False)
        return sampled_points

# 灵活的rf
class retina_polar_rf(nn.Module):
    '''
    Log polar transformation
    '''

    def __init__(
            self,
            r_min=0.05,
            r_max=0.8,
            H=5,
            W=12,
            log_r=True,
            retina_field=1,
    ):
        super(retina_polar_rf, self).__init__()
        if log_r:
            dr = (np.log(r_max * H) - np.log(r_min * H)) / H
        else:
            dr = ((r_max * H) - (r_min * H)) / H

        grid_2d = torch.empty(
            [H, W, 2]
        )
        angles = torch.empty(
            [H, W, 1]
        )
        for h in range(H):
            radius = r_min*h/H*np.exp(dr*h) # 这里，如果不要 h/H，就没有中间放大的效果了
            for w in range(W):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.H = H
        self.W = W
        self.retina_field = retina_field
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)

    def get_grid(self, b):
        radius = self.radius[None].clone().repeat(b, 1, 1, 1)
        angles = self.angles[None].clone().repeat(b, 1, 1, 1)
        angle = angles

        grid = torch.zeros_like(radius).to(device=radius.device)
        grid[:, :, :, 0] = radius[:, :, :, 0] * torch.sin(angle[:, :, :, 0])
        grid[:, :, :, 1] = radius[:, :, :, 1] * torch.cos(angle[:, :, :, 0])
        return grid

    def forward(self, x, l_t_prev, rf):
        batch_size, *_ = x.shape
        l_t_prev = l_t_prev.to(x.device)
        grid_2d_batch = self.get_grid(batch_size) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch * rf, padding_mode='border', align_corners=False)
        return sampled_points

class retina_polar_2rf(nn.Module):
    '''
    Log polar transformation
    '''

    def __init__(
            self,
            r_min=0.05,
            r_max=0.8,
            H=5,
            W=12,
            log_r=True,
    ):
        super(retina_polar_2rf, self).__init__()
        if log_r:
            dr = (np.log(r_max * H) - np.log(r_min * H)) / H
        else:
            dr = ((r_max * H) - (r_min * H)) / H

        grid_2d = torch.empty(
            [H, W, 2]
        )
        for h in range(H):
            radius = r_min*h/H*np.exp(dr*h) # 这里，如果不要 h/H，就没有中间放大的效果了
            for w in range(W):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    [radius * np.sin(angle), radius * np.cos(angle)]
                )
        self.H = H
        self.W = W
        self.register_buffer("grid_2d", grid_2d)

    def forward(self, x, l_t_prev, rf):
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) + self.grid_2d[None]
        grid_2d_batch = torch.stack([
            grid_2d_batch[:, :, :, 0] * rf[0],  # 高
            grid_2d_batch[:, :, :, 1] * rf[1]   # 宽
        ], dim=-1)
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border', align_corners=False)
        return sampled_points

class retina_polar_2rf_scale(nn.Module):
    """
    包含 scale和rotation的参数
    """

    def __init__(
            self,
            r_min=0.05,
            r_max=0.8,
            H=5,
            W=12,
            log_r=True,
            w_scale=8,
    ):
        super(retina_polar_2rf_scale, self).__init__()
        if log_r:
            dr = (np.log(r_max * H) - np.log(r_min * H)) / H
        else:
            dr = ((r_max * H) - (r_min * H)) / H

        grid_2d = torch.empty(
            [H, W, 2]
        )
        for h in range(H):
            radius = r_min*h/H*np.exp(dr*h) # 这里，如果不要 h/H，就没有中间放大的效果了
            for w in range(W):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    [radius * np.sin(angle), radius * np.cos(angle)]
                )
        self.H = H
        self.W = W
        self.register_buffer("grid_2d", grid_2d)
        self.w_scale = w_scale
    def forward(self, x, l_t_prev, rf, weight_s):
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) + self.grid_2d[None] 
        
        
        
        weight_s_unsqueezed = weight_s.unsqueeze(1)
        grid_2d_batch = torch.stack([
            grid_2d_batch[:, :, :, 0] * weight_s_unsqueezed * rf[0] * self.w_scale,  # 高
            grid_2d_batch[:, :, :, 1] * weight_s_unsqueezed * rf[1] * self.w_scale   # 宽
        ], dim=-1)
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border', align_corners=False)
        return sampled_points

class inverse_retina_polar(nn.Module):
    def __init__(
            self,
            r_min=0.01,
            r_max=0.6,
            retinal_H=5,
            retinal_W=5,
            H=5,
            W=12,
    ):
        super(inverse_retina_polar, self).__init__()
        self.H = H
        self.W = W
        self.r_min = r_min
        self.r_max = r_max
        grid_2d = torch.empty(
            [H, W, 2]
        )
        for i in range(H):
            for j in range(W):
                x = (i - int(H / 2)) / (H / 2)  # 这里除以H/2的依据是，r轴长H/2就覆盖了整个面积，然后归一化
                y = (j - int(W / 2)) / (W / 2)
                r = retinal_H * (np.log(np.clip(np.sqrt(x ** 2 + y ** 2), 1e-6, (H ** 2 + W ** 2))) - np.log(r_min)) / (
                        np.log(r_max) - np.log(r_min))
                a = np.arctan2(y, x)
                a = a if a > 0 else 2.0 * np.pi + a
                t = 0.5 * a * retinal_W / np.pi
                grid_2d[i, j] = torch.Tensor(
                    [t / (retinal_W / 2) - 1, r / (retinal_H / 2) - 1]
                )
        self.register_buffer("grid_2d", grid_2d)

    def forward(self, x, l_t_prev=torch.Tensor([0,0])):
        l_t_prev = l_t_prev.to(x.device)
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) * 0 + self.grid_2d[None]
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border', align_corners=False)
        return sampled_points

class Retina_SR_iter_residual(nn.Module):
    def __init__(self,
                 channel=3,
                 r_min=0.01,
                 r_max=np.sqrt(2),
                 image_H=224,
                 image_W=224,
                 retinal_H=224,
                 retinal_W=224,
                 em=1,  # eye movement
                 sampling_model = 'gaussian',
                 w_scale=8,
                 ):
        super(Retina_SR_iter_residual, self).__init__()
        self.retina = retina_polar_2rf_scale(
            r_min=r_min,
            r_max=r_max,
            H=retinal_H,
            W=retinal_W,
            log_r=True,
            w_scale=w_scale,
        )
        self.inverse_retina = inverse_retina_polar(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
        )
        point_number = self.nearest_square_root(em)
        if sampling_model == 'uniform':
            random_x = 2 * torch.rand(int(np.sqrt(point_number))) - 1
            combinations = list(itertools.product(random_x, repeat=2))
            sample_points = combinations[:point_number]
            sample_points = torch.tensor(sample_points)
        elif sampling_model == 'gaussian':
            random_x = torch.randn(int(np.sqrt(point_number))) / 2
            random_x[random_x > 1] = 1
            random_x[random_x < -1] = -1
            combinations = list(itertools.product(random_x, repeat=2))
            sample_points = combinations[:point_number]
            sample_points = torch.tensor(sample_points)
        indices = torch.randperm(sample_points.size(0))[:em - 1]
        center = torch.tensor((0, 0)).unsqueeze(0)
        sample_points = torch.cat([center, sample_points[indices, :]], 0)
        self.l_t = sample_points
        self.retinal_H = retinal_H
        self.retinal_W = retinal_W
        self.em = em
    def nearest_square_root(self, num):
        nearest = int(math.sqrt(num))
        while nearest * nearest < num:
            nearest += 1
        return nearest * nearest
    def crop_residual_by_lt(self, x, l_t, out_h, out_w):
        """
        x:   (B, C, H, W)
        l_t: (B, 2), range [-1, 1]
        """
        B, C, H, W = x.shape
        device = x.device

        cx = ((l_t[:, 0] + 1) * 0.5 * (W - 1)).long()
        cy = ((l_t[:, 1] + 1) * 0.5 * (H - 1)).long()

        half_w = out_w // 2
        half_h = out_h // 2

        x_resi = torch.zeros((B, C, out_h, out_w), device=device)

        for i in range(B):
            x1 = torch.clamp(cx[i] - half_w, 0, W - out_w)
            y1 = torch.clamp(cy[i] - half_h, 0, H - out_h)

            x_resi[i] = x[i, :, y1:y1 + out_h, x1:x1 + out_w]

        return x_resi

    def forward(self, x, l_t=None, s_t=None):
        if l_t == None:
            l_t = self.l_t
        b, c, h, w = x.shape
        x_retina = self.retina(x.unsqueeze(1).repeat(1, self.em, 1, 1, 1).reshape(-1, c, h, w),l_t, (1,1), s_t)
        x_resi = F.interpolate(x, size=(self.retinal_H, self.retinal_W), mode='bilinear', align_corners=False)
        
        x = self.inverse_retina(x_retina.reshape(-1, c, self.retinal_H, self.retinal_W),torch.zeros(b*self.em,1,2).to(x.device))
        return torch.cat([x,x_resi], 1), l_t, torch.tensor([1, 1]).repeat(b,1).float().to(x.device)
