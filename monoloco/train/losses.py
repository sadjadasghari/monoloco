"""Inspired by Openpifpaf"""

import math
import torch
import torch.nn as nn
import numpy as np

from ..network import extract_labels, extract_outputs


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, losses, lambdas, tasks):
        super().__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.tasks = tasks

    def forward(self, outputs, labels):

        out = extract_outputs(outputs, tasks=self.tasks)
        gt_out = extract_labels(labels, tasks=self.tasks)
        loss_values = [lam * l(o, g) for lam, l, o, g in zip(self.lambdas, self.losses, out, gt_out)]
        loss = sum(loss_values)

        return loss


class CompositeLoss(torch.nn.Module):

    def __init__(self, tasks):
        super(CompositeLoss, self).__init__()

        self.tasks = tasks
        self.multi_loss = {task: (LaplacianLoss() if task == 'loc' else nn.L1Loss()) for task in tasks}

    def forward(self):
        losses = [self.multi_loss[l] for l in self.tasks]
        return losses


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
