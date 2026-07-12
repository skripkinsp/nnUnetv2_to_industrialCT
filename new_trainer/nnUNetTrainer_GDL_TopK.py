import torch
from torch import nn
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import torch.nn.functional as F

class SafeGeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, max_weight=1000.0):
        super().__init__()
        self.smooth = smooth
        self.max_weight = max_weight # Жесткий клиппинг против NaN при FP16
        self.weights = torch.tensor([1.0, 3.0, 1.0, 3.0], dtype=torch.float)

    def forward(self, net_output, target):
        net_output = torch.softmax(net_output, dim=1)
        axes = tuple(range(2, net_output.ndim))

        with torch.no_grad():
            target_onehot = torch.zeros_like(net_output)
            target_onehot.scatter_(1, target.long(), 1)

        class_volumes = torch.sum(target_onehot, dim=axes)
        volumes_squared = class_volumes ** 2

        # Безопасный расчет w_l = 1 / V^2
        weights = torch.zeros_like(class_volumes)
        valid_mask = class_volumes > 0
        weights[valid_mask] = 1.0 / volumes_squared[valid_mask]
        weights = torch.clamp(weights, min=0.0, max=self.max_weight)

        intersection = torch.sum(net_output * target_onehot, dim=axes)
        union = torch.sum(net_output + target_onehot, dim=axes)

        weighted_intersection = torch.sum(weights * intersection, dim=1)
        weighted_union = torch.sum(weights * union, dim=1)

        gdl = 2.0 * (weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        return 1.0 - gdl.mean()

class TopK_CE_Loss(nn.Module):
    def __init__(self, k_percent=0.1):
        super().__init__()
        self.k_percent = k_percent
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, net_output, target):
        target_ce = target[:, 0].long()
        loss_matrix = self.ce(net_output, target_ce)
        loss_flat = loss_matrix.view(-1)
        
        k_pixels = int(self.k_percent * loss_flat.numel())
        if k_pixels == 0:
            return loss_flat.mean()
            
        topk_loss, _ = torch.topk(loss_flat, k_pixels)
        return topk_loss.mean()

class LightweightSurfaceLoss(nn.Module):
    def __init__(self, theta=3):
        super().__init__()
        self.theta = theta

    def get_boundary(self, x):
        dilated = F.max_pool3d(
            x,
            kernel_size=self.theta,
            stride=1,
            padding=self.theta // 2
        )

        eroded = -F.max_pool3d(
            -x,
            kernel_size=self.theta,
            stride=1,
            padding=self.theta // 2
        )

        boundary = dilated - eroded
        return boundary

    def forward(self, net_output, target):
        probs = torch.softmax(net_output, dim=1)

        with torch.no_grad():
            target_onehot = torch.zeros_like(probs)
            target_onehot.scatter_(1, target.long(), 1)

        pred_boundary = self.get_boundary(probs)
        gt_boundary = self.get_boundary(target_onehot)

        loss = torch.abs(pred_boundary - gt_boundary)

        return loss.mean()

class GDL_TopK_Surface_Loss(nn.Module):
    def __init__(
        self,
        weight_ce=1.0,
        weight_dice=1.0,
        weight_surface=0.15,
        k_percent=0.1,
        max_weight=1000.0
    ):
        super().__init__()

        self.ce = TopK_CE_Loss(k_percent)
        self.dice = SafeGeneralizedDiceLoss(max_weight=max_weight)
        self.surface = LightweightSurfaceLoss()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_surface = weight_surface

    def forward(self, net_output, target):
        ce = self.ce(net_output, target)
        dice = self.dice(net_output, target)
        surf = self.surface(net_output, target)

        return (
            self.weight_ce * ce +
            self.weight_dice * dice +
            self.weight_surface * surf
        )

class nnUNetTrainer_GDL_TopK(nnUNetTrainer):
    def build_loss(self):
        loss = GDL_TopK_Surface_Loss(
            weight_ce=1.0, 
            weight_dice=1.0, 
            weight_surface=0.3,
            k_percent=0.1,       # Интегрирование по 10% худших вокселей
            max_weight=1000.0    # Лимит GDL
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss