# pylint: disable=too-many-statements
"""
Training and evaluation of a neural network which predicts 3D localization and confidence intervals
given 2d joints
"""

import copy
import os
import datetime
import logging
from collections import defaultdict
import sys
import time
import warnings
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .datasets import KeypointsDataset
from ..network import LaplacianLoss, unnormalize_bi
from ..network.architectures import LinearModel
from ..utils import set_logger


class Trainer:
    # Constants
    INPUT_SIZE = 34
    OUTPUT_SIZE = 9
    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1

    lambd_ori = 2
    lambd_wlh = 0.2
    lambd_xy = 0.5

    def __init__(self, joints, epochs=100, bs=256, dropout=0.2, lr=0.002,
                 sched_step=20, sched_gamma=1, hidden_size=256, n_stage=3, r_seed=1, n_samples=100,
                 baseline=False, save=False, print_loss=True):
        """
        Initialize directories, load the data and parameters for the training
        """

        # Initialize directories and parameters
        dir_out = os.path.join('data', 'models')
        if not os.path.exists(dir_out):
            warnings.warn("Warning: output directory not found, the model will not be saved")
        dir_logs = os.path.join('data', 'logs')
        if not os.path.exists(dir_logs):
            warnings.warn("Warning: default logs directory not found")
        assert os.path.exists(joints), "Input file not found"

        self.joints = joints
        self.num_epochs = epochs
        self.save = save
        self.print_loss = print_loss
        self.baseline = baseline
        self.lr = lr
        self.sched_step = sched_step
        self.sched_gamma = sched_gamma
        self.clusters = ['10', '20', '30', '>30']
        self.hidden_size = hidden_size
        self.n_stage = n_stage
        self.dir_out = dir_out
        self.n_samples = n_samples
        self.r_seed = r_seed
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        name_out = 'monoloco-' + now_time

        # Select the device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:1" if use_cuda else "cpu")
        print('Device: ', self.device)
        torch.manual_seed(r_seed)
        if use_cuda:
            torch.cuda.manual_seed(r_seed)

        # Loss functions
        self.l_loc = LaplacianLoss().to(self.device)
        self.l_xy = nn.L1Loss().to(self.device)
        self.l_ori = nn.L1Loss().to(self.device)
        self.l_wlh = nn.L1Loss().to(self.device)
        self.l_eval = nn.L1Loss().to(self.device)
        self.C_laplace = torch.tensor([2.]).to(self.device)

        if self.save:
            self.path_model = os.path.join(dir_out, name_out + '.pkl')
            self.logger = set_logger(os.path.join(dir_logs, name_out))
            self.logger.info("Training arguments: \nepochs: {} \nbatch_size: {} \ndropout: {}"
                             "\nbaseline: {} \nlearning rate: {} \nscheduler step: {} \nscheduler gamma: {}  "
                             "\ninput_size: {} \noutput_size: {}\nhidden_size: {} \nn_stages: {} \nr_seed: {}"
                             "\ninput_file: {}\nlambda_ori: {}\nlambda_xy: {}\nlambda_wlh: {}"
                             .format(epochs, bs, dropout, baseline, lr, sched_step, sched_gamma, self.INPUT_SIZE,
                                     self.OUTPUT_SIZE, hidden_size, n_stage, r_seed, self.joints,
                                     self.lambd_ori, self.lambd_xy, self.lambd_wlh))
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        # Dataloader
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.joints, phase=phase),
                                              batch_size=bs, shuffle=True) for phase in ['train', 'val']}

        self.dataset_sizes = {phase: len(KeypointsDataset(self.joints, phase=phase))
                              for phase in ['train', 'val']}

        # Define the model
        self.logger.info('Sizes of the dataset: {}'.format(self.dataset_sizes))
        print(">>> creating model")
        self.model = LinearModel(input_size=self.INPUT_SIZE, output_size=self.OUTPUT_SIZE, linear_size=hidden_size,
                                 p_dropout=dropout, num_stage=self.n_stage)
        self.model.to(self.device)
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def train(self):

        # Initialize the variable containing model weights
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 1e6
        best_training_acc = 1e6
        best_epoch = 0
        epoch_losses = defaultdict(lambda: defaultdict(list))
        for epoch in range(self.num_epochs):
            running_loss = defaultdict(lambda: defaultdict(int))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    gt_dd = labels[:, 0:1]
                    gt_xy = labels[:, 1:3]
                    gt_loc = labels[:, 3:4]
                    gt_wlh = labels[:, 4:7]
                    gt_ori = labels[:, 7:9]

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        xy = outputs[:, 0:2]
                        loc = outputs[:, 2:4]
                        wlh = outputs[:, 4:7]
                        ori = outputs[:, 7:]
                        dd = torch.norm(outputs[:, 0:3], p=2, dim=1).view(-1, 1)

                        # backward + optimize only if in training phase
                        loss = \
                            self.l_loc(loc, gt_loc) + self.C_laplace + \
                            self.lambd_wlh * self.l_wlh(wlh, gt_wlh) + \
                            self.lambd_ori * self.l_ori(ori, gt_ori) + \
                            self.lambd_xy * self.l_ori(xy, gt_xy)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            running_loss['train']['all'] += loss.item() * inputs.size(0)
                            running_loss['train']['loc'] += \
                                (self.l_loc(loc, gt_loc).item() + self.C_laplace.item()) * inputs.size(0)
                            running_loss['train']['xy'] += \
                                self.lambd_xy * self.l_xy(xy, gt_xy).item() * inputs.size(0)
                            running_loss['train']['ori'] += \
                                self.lambd_ori * self.l_ori(ori, gt_ori).item() * inputs.size(0)
                            running_loss['train']['wlh'] += \
                                self.lambd_wlh * self.l_wlh(wlh, gt_wlh).item() * inputs.size(0) * self.WLH_STD
                        else:
                            running_loss['val']['all'] += loss.item() * inputs.size(0)
                            running_loss['val']['ori'] += get_angle_loss(ori, gt_ori).item() * inputs.size(0)
                            running_loss['val']['loc'] += self.l_eval(dd, gt_dd).item() * inputs.size(0)
                            running_loss['val']['xy'] += self.l_xy(xy, gt_xy).item() * inputs.size(0)
                            running_loss['val']['wlh'] += \
                                self.l_eval(wlh, gt_wlh).item() * inputs.size(0) * self.WLH_STD

            for phase in running_loss:
                for el in running_loss['train']:
                    epoch_losses[phase][el].append(running_loss[phase][el] / self.dataset_sizes[phase])

            if epoch % 5 == 0:
                sys.stdout.write('\r' + 'Epoch: {:.0f} '
                                        'Train: ALL: {:.2f}  Z: {:.2f} XY: {:.1f} Ori: {:.2f}  Wlh: {:.2f}    '
                                        'Val: ALL: {:.2f}  D: {:.2f}, XY: {:.2f}  Ori: {:.2f} WLh: {:.2f}'
                                 .format(
                    epoch, epoch_losses['train']['all'][-1], epoch_losses['train']['loc'][-1],
                    epoch_losses['train']['xy'][-1], epoch_losses['train']['ori'][-1],
                    epoch_losses['train']['wlh'][-1], epoch_losses['val']['all'][-1],
                    epoch_losses['val']['loc'][-1], epoch_losses['val']['xy'][-1],
                    epoch_losses['val']['ori'][-1], epoch_losses['val']['wlh'][-1]) + '\t')

            # deep copy the model
            if epoch_losses['val']['all'][-1] < best_acc:
                best_acc = epoch_losses['val']['all'][-1]
                best_training_acc = epoch_losses['train']['all'][-1]
                best_epoch = epoch
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('\n\n' + '-' * 120)
        self.logger.info('Training:\nTraining complete in {:.0f}m {:.0f}s'
                         .format(time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best training Accuracy: {:.3f}'.format(best_training_acc))
        self.logger.info('Best validation Accuracy: {:.3f}'.format(best_acc))
        self.logger.info('Saved weights of the model at epoch: {}'.format(best_epoch))

        if self.print_loss:
            print_losses(epoch_losses)
        # load best model weights
        self.model.load_state_dict(best_model_wts)

        return best_epoch

    def evaluate(self, load=False, model=None, debug=False):

        # To load a model instead of using the trained one
        if load:
            self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

        # Average distance on training and test set after unnormalizing
        self.model.eval()
        dic_err = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # initialized to zero
        phase = 'val'
        batch_size = 5000
        dataset = KeypointsDataset(self.joints, phase=phase)
        size_eval = len(dataset)
        start = 0
        with torch.no_grad():
            for end in range(batch_size, size_eval + batch_size, batch_size):
                end = end if end < size_eval else size_eval
                inputs, labels, _, _ = dataset[start:end]
                start = end
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Debug plot for input-output distributions
                if debug:
                    debug_plots(inputs, labels)
                    sys.exit()

                # Forward pass
                outputs = self.model(inputs)
                dic_err[phase]['all'] = self.compute_stats(outputs, labels, dic_err[phase]['all'], size_eval)

            print('-' * 120)
            self.logger.info("Evaluation, validation set: \nAverage distance D: {:.2f} m with bi {:.2f},  "
                             "xx: {:.2f} m \nAverage orientation: {:.1f} degrees \nAverage dimensions error: {:.0f} cm"
                             .format(dic_err['val']['all']['dd'], dic_err['val']['all']['bi'],
                                     dic_err['val']['all']['xy'],
                                     dic_err['val']['all']['ori'], dic_err['val']['all']['wlh'] * 100))

            # Evaluate performances on different clusters and save statistics
            for clst in self.clusters:
                inputs, labels, size_eval = dataset.get_cluster_annotations(clst)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass on each cluster
                outputs = self.model(inputs)
                # outputs = unnormalize_bi(outputs)

                dic_err[phase][clst] = self.compute_stats(outputs, labels, dic_err[phase][clst], size_eval)

                self.logger.info("{} errors in cluster {} --> D: {:.2f} m with a bi {:.1f},  XY: {:.2f} m "
                                 "Ori: {:.1f} degrees WLH: {:.0f} cm for {} instances. "
                                 .format(phase, clst, dic_err[phase][clst]['dd'], dic_err[phase][clst]['bi'],
                                         dic_err[phase][clst]['xy'], dic_err[phase][clst]['ori'],
                                         dic_err[phase][clst]['wlh'] * 100, size_eval))

        # Save the model and the results
        if self.save and not load:
            torch.save(self.model.state_dict(), self.path_model)
            print('-' * 120)
            self.logger.info("\nmodel saved: {} \n".format(self.path_model))
        else:
            self.logger.info("\nmodel not saved\n")

        return dic_err, self.model

    def compute_stats(self, outputs, labels, dic_err, size_eval):
        """Compute mean, bi and max of torch tensors"""

        gt_dd = labels[:, 0:1]
        gt_xy = labels[:, 1:3]
        gt_loc = labels[:, 3:4]
        gt_wlh = labels[:, 4:7]
        gt_ori = labels[:, 7:9]
        xy = outputs[:, 0:2]
        loc = outputs[:, 2:4]
        wlh = outputs[:, 4:7]
        ori = outputs[:, 7:]
        dd = torch.norm(outputs[:, 0:3], p=2, dim=1).view(-1, 1)
        bi = unnormalize_bi(outputs)
        mean_loc = float(self.l_eval(loc[:, 0:1], gt_loc).item())
        mean_dd = float(self.l_eval(dd, gt_dd).item())
        mean_bi = float(torch.mean(bi).item())
        mean_xy = float(self.l_eval(xy, gt_xy).item())
        mean_ori = float(get_angle_loss(ori, gt_ori).item())
        mean_wlh = float(self.l_eval(wlh, gt_wlh).item())

        dic_err['loc'] += mean_loc * (outputs.size(0) / size_eval)
        dic_err['xy'] += mean_xy * (outputs.size(0) / size_eval)
        dic_err['dd'] += mean_dd * (outputs.size(0) / size_eval)
        dic_err['bi'] += mean_bi * (outputs.size(0) / size_eval)
        dic_err['ori'] += mean_ori * (outputs.size(0) / size_eval)
        dic_err['wlh'] += mean_wlh * (outputs.size(0) / size_eval) * self.WLH_STD
        dic_err['count'] += (outputs.size(0) / size_eval)
        return dic_err


def debug_plots(inputs, labels):
    inputs_shoulder = inputs.cpu().numpy()[:, 5]
    inputs_hip = inputs.cpu().numpy()[:, 11]
    labels = labels.cpu().numpy()
    heights = inputs_hip - inputs_shoulder
    plt.figure(1)
    plt.hist(heights, bins='auto')
    plt.show()
    plt.figure(2)
    plt.hist(labels, bins='auto')
    plt.show()


def get_angle_loss(orient, gt_orient):
    angle = torch.atan2(orient[:, 0], orient[:, 1]) * 180 / math.pi
    try:
        gt_angle = torch.atan2(gt_orient[:, 0], gt_orient[:, 1]) * 180 / math.pi
    except IndexError:
        aa = 4
    angle_loss = torch.mean(torch.abs(angle - gt_angle))
    return angle_loss


def print_losses(epoch_losses):
    for idx, phase in enumerate(epoch_losses):
        for idx_2, el in enumerate(epoch_losses['train']):
            plt.figure(idx + idx_2)
            plt.plot(epoch_losses[phase][el][10:], label='{} Loss: {}'.format(phase, el))
            plt.savefig('figures/{}_loss_{}.png'.format(phase, el))
            plt.close()
