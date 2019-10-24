
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
from itertools import chain

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .datasets import KeypointsDataset
from .losses import CompositeLoss, MultiTaskLoss, AutoTuneMultiTaskLoss
from ..network import extract_outputs, extract_labels
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
    VAL_BS = 5000

    tasks = ('loc', 'xy', 'wlh', 'ori')
    # lambdas = (1, 0.5, 0.2, 2)
    lambdas = (1., 1., 1., 1.)

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
        self.auto_tune_mtl = True

        # Select the device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:1" if use_cuda else "cpu")
        print('Device: ', self.device)
        torch.manual_seed(r_seed)
        if use_cuda:
            torch.cuda.manual_seed(r_seed)

        losses_tr, losses_val = CompositeLoss(self.tasks)()

        if self.auto_tune_mtl:
            self.mt_loss = AutoTuneMultiTaskLoss(losses_tr, losses_val, self.lambdas, self.tasks)
        else:
            self.mt_loss = MultiTaskLoss(losses_tr, losses_val, self.lambdas, self.tasks)
        self.mt_loss.to(self.device)

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        name_out = 'monoloco-' + now_time
        if self.save:
            self.path_model = os.path.join(dir_out, name_out + '.pkl')
            self.logger = set_logger(os.path.join(dir_logs, name_out))
            self.logger.info("Training arguments: \nepochs: {} \nbatch_size: {} \ndropout: {}"
                             "\nbaseline: {} \nlearning rate: {} \nscheduler step: {} \nscheduler gamma: {}  "
                             "\ninput_size: {} \noutput_size: {}\nhidden_size: {} \nn_stages: {} \nr_seed: {}"
                             "\ninput_file: {}\nlambda_xy: {}\nlambda_wlh: {}\nlambda_ori: {}"
                             .format(epochs, bs, dropout, baseline, lr, sched_step, sched_gamma, self.INPUT_SIZE,
                                     self.OUTPUT_SIZE, hidden_size, n_stage, r_seed, self.joints,
                                     self.lambdas[1], self.lambdas[2], self.lambdas[3]))
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
        print(">>> model params: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        print(">>> loss params: {}".format(sum(p.numel() for p in self.mt_loss.parameters())))
        # Optimizer and scheduler
        all_params = chain(self.model.parameters(), self.mt_loss.parameters())
        self.optimizer = torch.optim.Adam(params=all_params, lr=lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step, gamma=self.sched_gamma)

    def train(self):

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

                for inputs, labels, _, _ in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    loss, loss_values = self.mt_loss(outputs, labels, phase=phase)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    self.epoch_logs(phase, loss, loss_values, inputs, running_loss, epoch_losses)

            self.cout_values(epoch, epoch_losses)

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

    def epoch_logs(self, phase, loss, loss_values, inputs, running_loss, epoch_losses):

        running_loss[phase]['all'] += loss.item() * inputs.size(0)
        for i, task in enumerate(self.tasks):
            running_loss[phase][task] += loss_values[i].item() * inputs.size(0)

        for phase in running_loss:
            for el in running_loss['train']:
                epoch_losses[phase][el].append(running_loss[phase][el] / self.dataset_sizes[phase])

    def evaluate(self, load=False, model=None, debug=False):

            # To load a model instead of using the trained one
            if load:
                self.model.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

            # Average distance on training and test set after unnormalizing
            self.model.eval()
            dic_err = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # initialized to zero
            dic_err['val']['sigmas'] = [0.] * len(self.tasks)
            dataset = KeypointsDataset(self.joints, phase='val')
            size_eval = len(dataset)
            start = 0
            with torch.no_grad():
                for end in range(self.VAL_BS, size_eval + self.VAL_BS, self.VAL_BS):
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
                    self.compute_stats(outputs, labels, dic_err['val'], size_eval, clst='all')

                self.cout_stats(dic_err['val'], size_eval, clst='all')
                # Evaluate performances on different clusters and save statistics
                for clst in self.clusters:
                    inputs, labels, size_eval = dataset.get_cluster_annotations(clst)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass on each cluster
                    outputs = self.model(inputs)
                    self.compute_stats(outputs, labels, dic_err['val'], size_eval, clst=clst)
                    self.cout_stats(dic_err['val'], size_eval, clst=clst)

            # Save the model and the results
            if self.save and not load:
                torch.save(self.model.state_dict(), self.path_model)
                print('-' * 120)
                self.logger.info("\nmodel saved: {} \n".format(self.path_model))
            else:
                self.logger.info("\nmodel not saved\n")

            return dic_err, self.model

    def compute_stats(self, outputs, labels, dic_err, size_eval, clst):
        """Compute mean, bi and max of torch tensors"""

        loss, loss_values = self.mt_loss(outputs, labels, phase='val')
        rel_frac = outputs.size(0) / size_eval

        for idx, task in enumerate(self.tasks):
            dic_err[clst][task] += float(loss_values[idx].item()) * (outputs.size(0) / size_eval)

        err_dd = torch.mean(torch.abs(extract_outputs(outputs)['dd'] - extract_labels(labels)['dd'])).item()
        bi = float(torch.mean(extract_outputs(outputs)['bi']).item())
        dic_err[clst]['dd'] += err_dd * rel_frac
        dic_err[clst]['bi'] += bi * rel_frac
        dic_err[clst]['count'] += rel_frac
        if self.auto_tune_mtl:
            assert len(loss_values) == 2 * len(self.tasks)
            for i, _ in enumerate(self.tasks):
                dic_err['sigmas'][i] += float(loss_values[idx+i+1].item()) * rel_frac

    def cout_stats(self, dic_err, size_eval, clst):
        if clst == 'all':
            print('-' * 120)
            self.logger.info("Evaluation, validation set: \nAv. distance D: {:.2f} m with bi {:.2f},  "
                             "XY: {:.2f} m \nAv. orientation: {:.1f} degrees \nAv. dimensions error: {:.0f} cm"
                             .format(dic_err[clst]['dd'], dic_err[clst]['bi'], dic_err[clst]['xy'],
                                     dic_err[clst]['ori'], dic_err[clst]['wlh'] * 100 * self.WLH_STD))
            if self.auto_tune_mtl:
                self.logger.info("Learned Sigmas: loc: {:.2f}, XY: {:.2f}, WLH: {:.2f}, ORI: {:.2f}"
                                 .format(*dic_err['sigmas']))
        else:
            self.logger.info("Validation errors in cluster {} --> D: {:.2f} m with a bi {:.2f},  XY: {:.2f} m "
                             "Ori: {:.1f} degrees WLH: {:.0f} cm for {} instances. "
                             .format(clst, dic_err[clst]['dd'], dic_err[clst]['bi'], dic_err[clst]['xy'],
                                     dic_err[clst]['ori'], dic_err[clst]['wlh'] * 100 * self.WLH_STD, size_eval))

    def cout_values(self, epoch, epoch_losses):
        if epoch % 5 == 0:
            sys.stdout.write('\r' + 'Epoch: {:.0f} '
                                    'Train: ALL: {:.2f}  Z: {:.2f} XY: {:.1f} Ori: {:.2f}  Wlh: {:.2f}    '
                                    'Val: ALL: {:.2f}  D: {:.2f}, XY: {:.2f}  Ori: {:.2f} WLh: {:.2f}'
                             .format(epoch, epoch_losses['train']['all'][-1], epoch_losses['train']['loc'][-1],
                                     epoch_losses['train']['xy'][-1], epoch_losses['train']['ori'][-1],
                                     epoch_losses['train']['wlh'][-1] * self.WLH_STD, epoch_losses['val']['all'][-1],
                                     epoch_losses['val']['loc'][-1], epoch_losses['val']['xy'][-1],
                                     epoch_losses['val']['ori'][-1],
                                     epoch_losses['val']['wlh'][-1] * self.WLH_STD) + '\t')


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


def print_losses(epoch_losses):
    for idx, phase in enumerate(epoch_losses):
        for idx_2, el in enumerate(epoch_losses['train']):
            plt.figure(idx + idx_2)
            plt.plot(epoch_losses[phase][el][10:], label='{} Loss: {}'.format(phase, el))
            plt.savefig('figures/{}_loss_{}.png'.format(phase, el))
            plt.close()
