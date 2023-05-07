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
import numpy as np
from glob import glob
import torch
import torch.nn.parallel
import torch.nn as nn
from utils.utils import get_filename

# -----------------------------------------------------------------------------

class BaseTester():
    def __init__(self, args, out_dir):
        self.args = args
        self.out_dir = out_dir
        self.rendering_dir = os.path.join(out_dir, 'renderings')
        self.texture_dir = os.path.join(out_dir, 'textures')

    def initialize(self):
        self.control_seeds()
        self.mkdirs()
        self.register_mesh_sampler()
        self.register_model()
        self.resume_checkpoint()
        self.register_dataset()

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
        if not os.path.exists(self.model_dir):
            raise RuntimeError("Model directory not found: %s" % (self.model_dir))

        os.makedirs(self.out_dir, exist_ok=True)
        # os.makedirs(self.rendering_dir, exist_ok=True)
        # os.makedirs(self.texture_dir, exist_ok=True)

    # ----------------------
    # model

    def register_mesh_sampler(self):
        raise NotImplementedError(f"mesh sampler not registered")

    def register_model(self):
        raise NotImplementedError(f"model not yet registered")

    def set_eval(self):
        self.model.train(False)

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
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=False,
                                            worker_init_fn=self.worker_init_fn, **kwargs)

    def register_dataset(self):
        raise NotImplementedError(f"dataset not yet registered")

        # example
        self.dataset_train = None
        self.dataloader_train = None
        self.dataset_val = None
        self.dataloader_val = None

    # ----------------------
    # main components

    def feed_data(self, data, mode="train"):
        # consumes a data instance from data loader to reorganize for model.forward format
        # produces self.data and self.inputs as dict
        raise NotImplementedError(f"feed_data() not defined yet")

        # example
        self.data = data
        self.inputs = {
            # reorganize data here, should fit the model.forward() input args
        }

    def forward(self):
        # consumes self.inputs
        # produces self.predicted
        raise NotImplementedError(f"forward() not defined yet")

        # example
        stuff = self.model(**self.inputs) # imgs=imgs, RTs=RTs, Ks=Ks, random_grid=True
        self.predicted = {
            # save your stuff here
        }

    def run(self):
        pass
