"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import os
from os.path import join
import time
import math
import numpy as np
import argparse
import imageio
from glob import glob
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

from utils.utils import get_filename

# -----------------------------------------------------------------------------

class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.global_step = 0

    def initialize(self):
        self.control_seeds()
        self.mkdirs()
        self.register_mesh_sampler()
        self.register_model()
        self.register_losses()
        self.register_optimizer()
        self.resume_checkpoint()
        self.register_dataset()
        self.register_logger()
        self.register_visualizer()

    # ----------------------
    # meta

    def control_seeds(self):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)

    def mkdirs(self):
        if self.args.model_directory == '': raise RuntimeError(f'invalid model_directory = {self.args.model_directory}')
        if self.args.experiment_id == '': raise RuntimeError(f'invalid experiment_id = {self.args.experiment_id}')
        self.directory_output = join(self.args.model_directory, self.args.experiment_id)
        os.makedirs(self.directory_output, exist_ok=True)

        self.model_dir = join(self.directory_output, 'checkpoints')
        os.makedirs(self.model_dir, exist_ok=True)

    # ----------------------
    # model

    def register_mesh_sampler(self):
        raise NotImplementedError(f"mesh sampler not registered")

    def register_model(self):
        raise NotImplementedError(f"model not yet registered")

    def register_losses(self):
        raise NotImplementedError(f"model not yet registered")

    def set_train(self):
        self.model.train(True)

    def set_eval(self):
        self.model.train(False)

    def set_test(self):
        self.set_eval()

    def save_checkpoint(self):
        model_path = os.path.join(self.model_dir, 'model_%08d.pth' % (self.global_step))
        torch.save({
            'model': self.model.module.state_dict(),
            'optimizer_model': self.optimizer_model.state_dict()
            }, model_path)

    def resume_checkpoint(self): 
        model_paths = sorted(glob(join(self.model_dir, '*.pth')))
        print(f"resume_checkpoint(): found {len(model_paths)} models")
        if len(model_paths) > 0:
            # pick the latest one
            resume_path = model_paths[-1]
            start_iteration =  int(get_filename(resume_path)[6:])
            self.global_step = start_iteration+1

            # load
            try:
                state_dicts = torch.load(resume_path)
                self.model.module.load_state_dict(state_dicts['model'])
                self.optimizer_model.load_state_dict(state_dicts['optimizer_model'])
                print('Resuming progress from %s iteration' % self.global_step)
                print(f"\tfrom model path {resume_path}")
            except Exception as e:
                self.model.load(resume_path)
                # ignore loading optimizer info
                print('(WORKAROUND) Resuming progress from %s iteration' % self.global_step)
                print(f"\tfrom model path {resume_path}")

    # ----------------------
    # dataset

    def worker_init_fn(self, worker_id):
        # to properly randomize:
        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def make_data_loader(self, dataset, cuda=True, shuffle=True):
        kwargs = {'num_workers': self.args.thread_num, 'pin_memory': True} if cuda else {}
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=True,
                                            worker_init_fn=self.worker_init_fn, **kwargs)

    def register_dataset(self):
        raise NotImplementedError(f"dataset not yet registered")

        # example
        self.dataset_train = None
        self.dataloader_train = None
        self.dataset_val = None
        self.dataloader_val = None

    # ----------------------
    # optimizer

    def register_optimizer(self):
        base_params, special_params = [], []
        for name, param in self.model.named_parameters(): 
            if 'grid_refiner' in name:
                special_params.append(param)
            else:
                base_params.append(param)

        self.optimizer_model = torch.optim.AdamW([
                                    {'params': base_params, 'lr': self.args.learning_rate, 'group_id': 'base'}, 
                                    {'params': special_params, 'lr': 1e-4, 'group_id': 'special'}])        

    def adjust_learning_rate(self, i, n_i, method):
        if method == 'step':
            lr, decay = self.args.learning_rate, 0.5
            if i >= n_i * 0.8:
                lr *= decay**4     
            elif i >= n_i * 0.6:
                lr *= decay**3       
            elif i >= n_i * 0.4:
                lr *= decay**2                      
            elif i >= n_i * 0.2:
                lr *= decay                                               
        elif method == 'constant':
            lr = self.args.learning_rate
        elif method == 'exp':
            lr = self.args.learning_rate * (1 + self.args.lr_gamma * i) ** (-self.args.lr_power)
        elif method == 'cosine':
            lr = 0.5 * self.args.learning_rate * (1 + math.cos(math.pi * i / n_i))
        else:
            print("no such learing rate type")

        optimizers = [self.optimizer_model]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                if ('group_id' not in param_group) or (param_group['group_id'] == 'base'):
                    param_group['lr'] = lr
        return lr

    # ----------------------
    # logger

    def register_logger(self):
        # from utils.simple_logger import Logger
        from utils.tensorboard_logger import Logger as TensorboardLogger

        # tensorboard logger
        self.tb_logger = TensorboardLogger(os.path.join(self.directory_output, 'logs'))

    # ----------------------
    # visualizer

    def register_visualizer(self):
        self.visualizer = None

    # ----------------------
    # main components

    def feed_data(self, data, mode="train"):
        # consumes a data instance from data loader to reorganize for model.forward format
        # produces self.data and self.inputs as dict
        raise NotImplementedError(f"feed_data() not defined yet")

    def forward(self):
        # consumes self.inputs
        # produces self.predicted
        raise NotImplementedError(f"forward() not defined yet")

    def compute_losses(self):
        # consumes self.inputs, self.data, self.predicted
        # produces self.loss (one single scalar loss to be optimized) and other losses during training
        # also records the losses
        raise NotImplementedError(f"compute_losses() not defined yet")

    def backward(self):
        raise NotImplementedError(f"backward() not defined yet")

    def save_visualizations(self, demo_id, mode='train'):
        # produces and saves visualization
        raise NotImplementedError(f"save_visualizations() not defined yet")

    # ----------------------
    # main processes

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def run(self):
        pass
