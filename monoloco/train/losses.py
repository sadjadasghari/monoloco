"""Inspired by Openpifpaf"""

import math
import torch
import torch.nn as nn
import numpy as np

from ..network import extract_labels, extract_outputs


class AutoTuneMultiTaskLoss(torch.nn.Module):
    def __init__(self, losses_tr, losses_val, lambdas, tasks):
        super().__init__()

        assert all(l in (0.0, 1.0) for l in lambdas)
        self.losses = torch.nn.ModuleList(losses_tr)
        self.losses_val = losses_val
        self.lambdas = lambdas
        self.tasks = tasks
        self.log_sigmas = torch.nn.Parameter(torch.zeros((len(lambdas),), dtype=torch.float32), requires_grad=True)

    def forward(self, outputs, labels, phase='train'):

        assert phase in ('train', 'val')
        out = extract_outputs(outputs, tasks=self.tasks)
        gt_out = extract_labels(labels, tasks=self.tasks)
        loss_values = [lam * l(o, g) / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l, o, g in zip(self.lambdas, self.log_sigmas, self.losses, out, gt_out)]

        auto_reg = [log_sigma for log_sigma in self.log_sigmas]

        loss = sum(loss_values) + sum(auto_reg)
        if phase == 'val':
            loss_values_val = [l(o, g) for l, o, g in zip(self.losses_val, out, gt_out)]
            loss_values_val.extend([s.exp() for s in self.log_sigmas])
            return loss, loss_values_val
        else:
            return loss, loss_values


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, losses_tr, losses_val, lambdas, tasks):
        super().__init__()

        self.losses = torch.nn.ModuleList(losses_tr)
        self.losses_val = losses_val
        self.lambdas = lambdas
        self.tasks = tasks

    def forward(self, outputs, labels, phase='train'):

        assert phase in ('train', 'val')
        out = extract_outputs(outputs, tasks=self.tasks)
        gt_out = extract_labels(labels, tasks=self.tasks)
        loss_values = [lam * l(o, g) for lam, l, o, g in zip(self.lambdas, self.losses, out, gt_out)]
        loss = sum(loss_values)
        if phase == 'val':
            loss_values_val = [l(o, g) for l, o, g in zip(self.losses_val, out, gt_out)]
            return loss, loss_values_val
        else:
            return loss, loss_values


class CompositeLoss(torch.nn.Module):

    def __init__(self, tasks):
        super(CompositeLoss, self).__init__()

        self.tasks = tasks
        self.multi_loss_tr = {task: (LaplacianLoss() if task == 'loc' else nn.L1Loss()) for task in tasks}
        self.multi_loss_val = {task: (l1_loss_from_laplace if task == 'loc'
                                      else (angle_loss if task == 'ori'
                                            else nn.L1Loss())) for task in tasks}

    def forward(self):
        losses_tr = [self.multi_loss_tr[l] for l in self.tasks]
        losses_val = [self.multi_loss_val[l] for l in self.tasks]
        return losses_tr, losses_val


class LaplacianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance"""
    def __init__(self, size_average=True, reduce=True, evaluate=False):
        super(LaplacianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate

    def laplacian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py

        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]
        # norm = xx - mu
        norm = 1 - mu / xx  # Relative

        term_a = torch.abs(norm) * torch.exp(-si)
        term_b = si
        norm_bi = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(torch.exp(si).cpu().detach().numpy()))

        if self.evaluate:
            return norm_bi
        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.laplacian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


class GaussianLoss(torch.nn.Module):
    """1D Gaussian with std depending on the absolute distance
    """
    def __init__(self, device, size_average=True, reduce=True, evaluate=False):
        super(GaussianLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.evaluate = evaluate
        self.device = device

    def gaussian_1d(self, mu_si, xx):
        """
        1D Gaussian Loss. f(x | mu, sigma). The network outputs mu and sigma. X is the ground truth distance.
        This supports backward().
        Inspired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        """
        mu, si = mu_si[:, 0:1], mu_si[:, 1:2]

        min_si = torch.ones(si.size()).cuda(self.device) * 0.1
        si = torch.max(min_si, si)
        norm = xx - mu
        term_a = (norm / si)**2 / 2
        term_b = torch.log(si * math.sqrt(2 * math.pi))

        norm_si = (np.mean(np.abs(norm.cpu().detach().numpy())), np.mean(si.cpu().detach().numpy()))

        if self.evaluate:
            return norm_si

        return term_a + term_b

    def forward(self, outputs, targets):

        values = self.gaussian_1d(outputs, targets)

        if not self.reduce or self.evaluate:
            return values
        if self.size_average:
            mean_values = torch.mean(values)
            return mean_values
        return torch.sum(values)


def angle_loss(orient, gt_orient):
    """Only for evaluation"""
    angle = torch.atan2(orient[:, 0], orient[:, 1]) * 180 / math.pi
    gt_angle = torch.atan2(gt_orient[:, 0], gt_orient[:, 1]) * 180 / math.pi
    angle_loss = torch.mean(torch.abs(angle - gt_angle))
    return angle_loss


def l1_loss_from_laplace(loc, gt_loc):
    """Only for evaluation"""
    loss = torch.mean(torch.abs(loc[:, 0:1] - gt_loc))
    return loss

