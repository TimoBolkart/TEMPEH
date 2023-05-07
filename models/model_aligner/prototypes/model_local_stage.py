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
import numpy as np
import torch
import torch.nn as nn
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, args, mesh_sampler, feature_net=None):
        super(Model, self).__init__()

        # ---- properties ----
        self.args = args
        self.input_ch = 3
        self.descriptor_dim = args.descriptor_dim

        self.local_voxel_dim = args.local_voxel_dim
        self.local_voxel_inc_list = args.local_voxel_inc_list
        self.norm = args.norm

        # ---- submodules ----
        self.module_names = ['feature_net', 'local_densify_net']

        # feature extractor for 2d
        if feature_net is not None:
            self.feature_net = feature_net
        else:
            from models.model_aligner import FeatureNet2D
            self.feature_net = FeatureNet2D(input_ch=self.input_ch, output_ch=self.descriptor_dim, architecture=args.feature_arch)

        local_feature_fusion = args.local_feature_fusion if hasattr(args, 'local_feature_fusion') else 'baseline_mean_var'

        # local upsample and refinement net
        from models.model_aligner import LocalDensifyNet
        self.local_densify_net = LocalDensifyNet(
                input_ch=self.descriptor_dim,
                mesh_sampler=mesh_sampler,
                local_architecture=args.local_arch,
                local_voxel_dim=self.local_voxel_dim,
                local_voxel_inc_list=self.local_voxel_inc_list,
                global_embedding_type=args.global_embedding_type,
                norm=self.norm,
                local_feature_fusion=local_feature_fusion)

    def forward(self, images, camera_intrinsics, camera_extrinsics, camera_distortions, camera_centers, global_points, random_grid=True):
        batch_size, num_images, num_channels, image_height, image_width = images.shape
        assert num_channels == self.input_ch, f"unmatched input image channel: {num_channels} (expected: {self.input_ch})"

        # step 1: feature extraction
        images = images.view(-1, num_channels, image_height, image_width)
        feature_maps = self.feature_net(images)

        feature_maps = feature_maps.view(batch_size, num_images, -1, image_height, image_width) # (B, V, F, H', W')

        # step 2: upsample and refinement
        densified_data = self.local_densify_net(
                                feature_maps=feature_maps, 
                                camera_intrinsics=camera_intrinsics, 
                                camera_extrinsics=camera_extrinsics, 
                                camera_distortions=camera_distortions, 
                                camera_centers=camera_centers,
                                global_points=global_points, 
                                random_grid=random_grid
        )
        
        return densified_data
