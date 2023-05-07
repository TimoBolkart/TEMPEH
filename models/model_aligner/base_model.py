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
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

# -----------------------------------------------------------------------------

class BaseModel(nn.Module):

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

        # ---- modules ----
        # by default, use "model" as the field name. you could of course customize.
        self.module_names = ['model']
        self.model = None # should be an instance of nn.Module

    def initialize(self, init_method='kaiming', model_path=None, verbose=False):
        """initialize parameters by certain distribution
        Args:
            init_method: allowed: 'normal', 'xavier', 'kaiming', 'orthogonal', 'nothing'
            model_path: path to pretrained model
            verbose: if print out info
        """
        from modules.module_utils import init_weights
        for name in self.module_names:
            init_weights(getattr(self, name), init_type=init_method, verbose=verbose)

        if model_path is not None and isinstance(model_path, str):
            self.load_special(model_path, verbose=verbose)

    def load_special(self, model_path, verbose=False):
        """load model from pretrained model of not exactly this class
        i.e. you would need to copy some pretrained weights to this class
        - you may also decide specific method given your setting, e.g. self.architecture
        - always called if model_path is string
        """

        pass

    def load(self, model_path, verbose=False):
        """load model from pretrained model of exactly this class (often called manually)
        """
        if verbose: print( "initializing from pretrained model..." )
        tic = time.time()

        try:
            state_dicts = torch.load(model_path)
            self.load_state_dict(state_dicts['model'], strict=False)
            if verbose: print( "initialized model from pretrained %s (%.1f sec)" % ( model_path, time.time()-tic ) )

        except:
            from modules.module_utils import copy_weights
            if model_path not in ['', None]:
                copy_weights(src_path=model_path, dst_net=self,
                    keywords=None,
                    name_maps=[
                        lambda dst: dst,
                        # lambda dst: 'sparse_point_net.' + dst.replace('densify_net.', '')
                    ], verbose=True)
                if verbose: print( "(SPECIAL) initialized model from pretrained %s (%.1f sec)" % ( model_path, time.time()-tic ) )

    def save(self, model_dir, iter_num):
        raise NotImplementedError

    def parms(self):
        parms_list = []
        for name in self.module_names:
            parms_list += list(getattr(self, name).parameters())
        return parms_list

    def optimizable_parms(self):
        """ parameters to be optimized. Default: all parameters
        This function can be override by child-classes
        """
        return self.parms()

    def named_parms(self):
        parms_dict = {}
        for name in self.module_names:
            parms_dict[name] = dict(getattr(self, name).named_parameters())
        return parms_dict

    def print(self, verbose=2):
        from utils.debug import print_network
        for name in self.module_names:
            print_network(getattr(self, name), verbose=verbose)

    def print_setting(self):
        # print out information as in **kwargs
        pass

    def forward(self, x):
        raise NotImplementedError(f"forward function not yet specified")
