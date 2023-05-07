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

    def __init__(self, args):
        super(Model, self).__init__()

        # ---- properties ----
        self.args = args
        self.input_ch = 3
        self.descriptor_dim = args.descriptor_dim
        self.number_points = min(args.number_sample_points)

        self.global_voxel_dim = args.global_voxel_dim
        self.global_voxel_inc = args.global_voxel_inc
        self.global_origin = args.global_origin
        self.norm = args.norm

        # ---- submodules ----
        self.module_names = ['feature_net', 'sparse_point_net']

        # feature extractor for 2d
        from models.model_aligner import FeatureNet2D
        self.feature_net = FeatureNet2D(
            input_ch=self.input_ch, output_ch=self.descriptor_dim, architecture=args.feature_arch)

        global_feature_fusion = args.global_feature_fusion if hasattr(args, 'global_feature_fusion') else 'baseline_mean_var'

        # global voxel net
        if args.global_spatial_transformer in [None, 'none', '']:
            from models.model_aligner import GlobalSparsePointNet
            self.sparse_point_net = GlobalSparsePointNet(
                input_ch=self.descriptor_dim, number_points=self.number_points, global_architecture=args.global_arch,
                global_voxel_dim=self.global_voxel_dim, global_voxel_inc=self.global_voxel_inc, global_origin=self.global_origin, 
                norm=self.norm, global_feature_fusion=global_feature_fusion)
        else:
            if args.global_spatial_transformer == 'rigid_transformer':
                transformer_type = 'rigid'
            elif args.global_spatial_transformer == 'rigid_transformer_res':
                transformer_type = 'rigid_res'
            elif args.global_spatial_transformer == 'rigid_transformer_deep_res':
                transformer_type = 'rigid_deep_res'                
            elif args.global_spatial_transformer == 'scale_trans_transformer':
                transformer_type = 'scale_trans'
            elif args.global_spatial_transformer == 'scale_trans_transformer_res':
                transformer_type = 'scale_trans_res'
            elif args.global_spatial_transformer == 'scale_trans_transformer_deep_res':
                transformer_type = 'scale_trans_deep_res'                
            else:
                raise RuntimeError("Unrecognizable global_spatial_transformer: %s" % (args.global_spatial_transformer))
            
            from models.model_aligner import GlobalSparsePointNetTransformer as GlobalSparsePointNet            
            self.sparse_point_net = GlobalSparsePointNet(
                input_ch=self.descriptor_dim, number_points=self.number_points, global_architecture=args.global_arch,
                global_voxel_dim=self.global_voxel_dim, global_voxel_inc=self.global_voxel_inc, global_origin=self.global_origin, 
                norm=self.norm, global_feature_fusion=global_feature_fusion, 
                transformer_type=transformer_type, num_transformer_channels=args.global_spatial_transformer_dim, num_transformer_levels=args.global_transformer_levels,
                transformer_scale_factor=args.global_transformer_scale_factor, transformer_use_pooling=args.global_transformer_use_pooling                
                )                

    def print_setting(self):
        pass

    def forward(self, images, camera_intrinsics, camera_extrinsics, camera_distortions,
                random_grid=True):

        # '''compute 2d feature maps given multiview images, and create plane sweep feature volume
        # Args:
        #     imgs: (B, V, 3, H', W'). H', W' are orig size, as compared to feature size in H, W
        #     RTs: (B, V, 3, 4)
        #     Ks: (B, V, 3, 3)
        # Returns:
        #     pts_global: (B, L, 3)
        #     pts_refined: (B, L, 3)
        #     global_cost_vol: (B, L, Dg, Dg, Dg)
        #     global_grid: (B, 1, 3, Dg, Dg, Dg)
        #     global_disp: (B, 1, 3, Dg, Dg, Dg)
        #     global_Rot: (B, 1, 3, 3)
        #     local_cost_vol: (B, L, D, D, D)
        #     local_grids: list of (B, L, 3, D, D, D)
        #     local_disps: list of (B, L, 3, D, D, D)
        #     local_Rot: (B, L, 3, 3)
        # '''

        batch_size, num_images, num_channels, image_height, image_width = images.shape
        assert num_channels == self.input_ch, f"unmatched input image channel: {num_channels} (expected: {self.input_ch})"

        images = images.view(-1, num_channels, image_height, image_width)
        feature_maps = self.feature_net(images)

        feature_maps = feature_maps.view(batch_size, num_images, -1, image_height, image_width)  # .view(bs, vn, -1, ih, iw) # (B, V, F, H', W')

        # step 2: sparse point prediction
        global_points, global_cost_vol, global_grid, global_grid_orign, global_grid_scales = self.sparse_point_net(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=random_grid)

        return global_points, global_cost_vol, global_grid, global_grid_orign, global_grid_scales