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

from glob import glob
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, input_ch, output_ch, architecture, **kwargs):
        super(Model, self).__init__()

        # ---- properties ----
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.architecture = architecture

        # ---- modules ----
        self.module_names = ['model']
        pretrained = True

        # feature extractor for 2d
        if self.architecture == 'uresnet':           # 21,3 Mio parameters
            import modules.resnet_dilated as resnet_dilated
            self.model = resnet_dilated.Resnet34_8s_skip(num_classes=self.output_ch, pretrained=pretrained) 
        elif self.architecture == 'uresnet2':       #  21,3 Mio parameters
            import modules.resnet_dilated as resnet_dilated
            self.model = resnet_dilated.Resnet34_8s_skip2(num_classes=self.output_ch, pretrained=pretrained)            
        elif self.architecture == 'unet_small1':    #   1,3 Mio parameters
            import modules.unet as unet
            size_factor = 1
            self.model = unet.GeneratorUNetSmall(in_channels=self.input_ch, out_channels=self.output_ch, size_factor=size_factor)
        elif self.architecture == 'unet_small2':    #   5,4 Mio parameters
            import modules.unet as unet
            size_factor = 2
            self.model = unet.GeneratorUNetSmall(in_channels=self.input_ch, out_channels=self.output_ch, size_factor=size_factor)        
        elif self.architecture == 'unet_small4':    #  21,4 Mio parameters
            import modules.unet as unet
            size_factor = 4
            self.model = unet.GeneratorUNetSmall(in_channels=self.input_ch, out_channels=self.output_ch, size_factor=size_factor) 
        elif self.architecture == 'convnext_tiny': #   28,6 Mio parameters
            import modules.convnext as convnext
            self.model = convnext.ConvNeXt_skip(in_channels=self.input_ch, out_channels=self.output_ch, architecture='tiny', pretrained=pretrained)
        elif self.architecture == 'convnext_small': #  50,2 Mio parameters
            import modules.convnext as convnext
            self.model = convnext.ConvNeXt_skip(in_channels=self.input_ch, out_channels=self.output_ch, architecture='small', pretrained=pretrained)
        elif self.architecture == 'convnext_base':  #  88,6 Mio parameters
            import modules.convnext as convnext
            self.model = convnext.ConvNeXt_skip(in_channels=self.input_ch, out_channels=self.output_ch, architecture='base', pretrained=pretrained)
        elif self.architecture == 'convnext_large': # 197,8 Mio parameters
            import modules.convnext as convnext
            self.model = convnext.ConvNeXt_skip(in_channels=self.input_ch, out_channels=self.output_ch, architecture='large', pretrained=pretrained)                        
        else:
            raise RuntimeError( "unrecognizable architecture: %s" % ( self.architecture ) )

    def load_special(self, model_path, verbose=False):
        if self.architecture == 'uresnet':
            # you may load the model accordingly for each architecture
            pass
        else:
            raise RuntimeError( "unrecognizable architecture: %s" % ( self.architecture ) )

    def print_setting(self):
        print("-"*40)
        print(f"name: feature_net_2d")
        print(f"\t- input_ch: {self.input_ch}")
        print(f"\t- output_ch: {self.output_ch}")
        print(f"\t- architecture: {self.architecture}")

    def forward(self, x):
        '''compute 2d feature maps given images
        Args:
            x: tensor in (B, C, H', W'). H', W' are orig size
        Returns:
            x: tensor in (B, F, H, W), note the height and width might change
        '''
        # meta
        bs, ic, ih, iw = x.shape
        device = x.device
        assert ic == self.input_ch, f"unmatched input image channel {ic}, expected {self.input_ch}"

        # run
        x = self.model(x)
        return x

