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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_aligner.base_model import BaseModel

# -----------------------------------------------------------------------------

class Model(BaseModel):

    def __init__(self, 
                input_ch, 
                number_points, 
                global_architecture='v2v',
                global_voxel_dim=32, 
                global_voxel_inc=1.0, 
                global_origin=[0., 0., 0.5], 
                norm='bn', 
                global_feature_fusion='',
                **kwargs):
        super(Model, self).__init__()

        # ---- properties ----
        self.input_ch = input_ch  # input feature dim
        self.number_points = number_points

        # global voxel setting
        self.global_architecture = global_architecture
        self.global_voxel_dim = global_voxel_dim
        self.global_voxel_inc = global_voxel_inc
        self.global_number_points = self.number_points
        self.global_feature_fusion = global_feature_fusion

        # note: if registered as buffer, then when loading from pretrain model, global_origin will also
        # be copied and thus the specified setting will be overrided.
        self.global_origin = torch.from_numpy(np.asarray(global_origin, dtype=np.float32))[None, None, :]  # (1,1,3)

        # volumetric feature sampler
        self.fuse_ops_num = 2  # hardcoded, same as len(VolumetricFeatureSampler.fuse_ops)

        self.norm = norm

        # ---- submodules ----
        self.module_names = ['global_net']

        if self.global_architecture == 'v2v2':
            from modules.v2v_coarse import V2VModel
            self.global_net = V2VModel(
                input_channels=self.fuse_ops_num*self.input_ch,
                output_channels=self.global_number_points,
                norm=self.norm)  # encdec_level=5                
        else:
            raise RuntimeError( "unrecognizable global_architecture: %s" % ( self.global_architecture ) )

        from modules.volumetric_feature_sampler import VolumetricFeatureSampler
        self.vfs = VolumetricFeatureSampler(feature_fusion=self.global_feature_fusion, feature_dim=self.input_ch)            

    def print_setting(self):
        print("-"*40)
        print(f"name: sparse_point_net_base")
        print(f"\t- input_ch: {self.input_ch}")
        print(f"\t- pts_num: {self.number_points}")
        print(f"\t- global_net:")
        print(f"\t\t- global_architecture: {self.global_architecture}")
        print(f"\t\t- global_voxel_dim: {self.global_voxel_dim}")
        print(f"\t\t- global_voxel_inc: {self.global_voxel_inc}")
        print(f"\t\t- global_pts_num: {self.global_number_points}")
        print(f"\t\t- global_origin: {self.global_origin}")
        print(f"\t\t- norm: {self.norm}")

    # ---- differentiable processes ----

    def normalize_volume(self, volume):
        ''' normalize volume spatially (throughout volume dimensions)
        Args and Returns:
            volume: (B, L, D, D, D)
        '''
        bs, ln, dim, dim, dim = volume.shape
        volume = torch.nn.Softmax(dim=2)(volume.view(bs, ln, -1)).view(bs, ln, dim, dim, dim)
        return volume

    def sample_global_features(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=False):
        '''sample volumetric features from global voxel grids
        Args:
            feats: tensor in (B, V, F, H, W), 2d features from images
            RTs: tensor in (B, V, 3, 4)
            Ks: tensor in (B, V, 3, 3)
        Returns:
            global_feat_grid: (B, 2F, Dg, Dg, Dg)
            global_grid: (B, 1, 3, Dg, Dg, Dg)
            global_disp: (B, 1, 3, Dg, Dg, Dg)
            global_Rot: (B, 1, 3, 3)
        '''
        
        bs, vn, fd, fh, fw = feature_maps.shape
        device = feature_maps.device
        gd = self.global_voxel_dim # Dg

        # create grid ((B,1,3,3), randomly rotated voxel grids)
        from modules.voxel_utils import sample_random_rotation, generate_local_grids
        global_Rot = sample_random_rotation(bs, 1).to(device) if random_grid else None
        global_grid, global_disp = generate_local_grids(
            vert=self.global_origin.to(device).repeat(bs,1,1),
            grid_dim=self.global_voxel_dim, # * 2,
            grid_inc=self.global_voxel_inc, # / 2.0,  # hack!
            rotate_mat=global_Rot)

        # sample volumetric features
        global_feat_grid = self.vfs.forward(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, global_grid)
        global_feat_grid = torch.squeeze(global_feat_grid, dim=1)
        return global_feat_grid, global_grid, global_disp, global_Rot

    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=True):
        ''' The global stage of ToFu (base), given images with known camera calibration,
        the network predicts coordinates of sparse initial vertices
        Args:
            feats: (B, V, F, H, W), 2d features from images
            RTs: (B, V, 3, 4)
            Ks: (B, V, 3, 3)
        Returns:
            pts_global: (B, L, 3)
            pts_refined: (B, L, 3)
            global_cost_vol: (B, L, Dg, Dg, Dg)
            global_grid: (B, 1, 3, Dg, Dg, Dg)
            transformed_grid_origin: 
        '''
        from modules.voxel_utils import compute_expectation
        bs, vn, fd, fh, fw = feature_maps.shape
        device = feature_maps.device
        gd = self.global_voxel_dim  # Dg
        ln = self.global_number_points  # L

        # sample features
        global_feature_grid, global_grid, global_disp, global_Rot = self.sample_global_features(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=random_grid)
        transformed_grid_origin = self.global_origin.squeeze().repeat(bs,1).to(device)

        # predict global points
        global_cost_volume = self.global_net(global_feature_grid)  # (B*1, L, Dg, Dg, Dg)
        global_cost_volume = self.normalize_volume(global_cost_volume)
        pts_global = compute_expectation(global_cost_volume, global_grid.repeat(1,ln,1,1,1,1))  # (B,L,3)

        return pts_global, global_cost_volume, global_grid, transformed_grid_origin, torch.ones(3).to(device) 