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
from modules.volumetric_feature_sampler import VolumetricFeatureSampler

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
                global_feature_fusion='mean_var',
                transformer_type='rigid',
                num_transformer_channels=32,
                num_transformer_levels=2,
                transformer_scale_factor=1.0,
                transformer_use_pooling=True,
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

        if transformer_type == 'rigid':
            from modules.grid_localizer import RigidGridLocalizer as GridLocalizer
            regress_rotation = True
        elif transformer_type == 'rigid_res':
            from modules.grid_localizer import RigidGridLocalizerResBlock as GridLocalizer
            regress_rotation = True
        elif transformer_type == 'rigid_deep_res':
            from modules.grid_localizer import DeepRigidGridLocalizerResBlock as GridLocalizer
            regress_rotation = True
        elif transformer_type == 'scale_trans':
            from modules.grid_localizer import RigidGridLocalizer as GridLocalizer
            regress_rotation = False
        elif transformer_type == 'scale_trans_res':
            from modules.grid_localizer import RigidGridLocalizerResBlock as GridLocalizer
            regress_rotation = False
        elif transformer_type == 'scale_trans_deep_res':
            from modules.grid_localizer import DeepRigidGridLocalizerResBlock as GridLocalizer
            regress_rotation = False
        else:
            raise RuntimeError( "Unrecognizable transformer_type: %s" % ( transformer_type ) )

        self.grid_refiner = GridLocalizer(input_channels=self.fuse_ops_num*self.input_ch, 
                                            volume_dim=self.global_voxel_dim, 
                                            hidden_channels=num_transformer_channels,
                                            num_levels=num_transformer_levels,
                                            regress_rotation=regress_rotation,
                                            scale_factor=transformer_scale_factor,
                                            pooling=transformer_use_pooling)                                        

        # self.debug_counter = 0

    def print_setting(self):
        pass
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
        volume = torch.nn.Softmax(dim=2)(
            volume.view(bs, ln, -1)).view(bs, ln, dim, dim, dim)
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
        # global_Rot = sample_random_rotation(bs, 1).to(device) if random_grid else None
        global_Rot = sample_random_rotation(bs, 1).to(device) if random_grid else (torch.eye(3)[None,None,:,:]).repeat(bs,1,1,1).to(device)
        global_grid, global_disp = generate_local_grids(
            vert=self.global_origin.to(device).repeat(bs,1,1),
            grid_dim=self.global_voxel_dim, 
            grid_inc=self.global_voxel_inc, 
            rotate_mat=global_Rot)

        # sample volumetric features
        global_feat_grid = self.vfs.forward(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, global_grid)
        global_feat_grid = torch.squeeze(global_feat_grid, dim=1)

        return global_feat_grid, global_grid, global_disp, global_Rot

    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=True):
        ''' The coarse stage of TEMPEH, given images with known camera calibration,
        the network predicts coordinates of sparse initial vertices
        Args:
            feature_maps: (B, V, F, H, W), 2d features from images
            camera_intrinsics:
            camera_extrinsics:
            camera_distortions:
            random_grid:

        Returns:
            pts_global: (B, L, 3)
            global_cost_volume: (B, L, Dg, Dg, Dg)
            warped_grid:
            transformed_grid_origin:
            grid_scales: 
        '''


        from modules.voxel_utils import compute_expectation
        device = feature_maps.device
        ln = self.global_number_points  # L

        # Sample features
        global_feature_grid, global_grid, global_disp, global_Rot = self.sample_global_features(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, random_grid=random_grid)
        bs, _, _, grid_h, grid_w, grid_d = global_grid.shape

        # Predict transformation to localize region of interest
        grid_transformation, grid_scales = self.grid_refiner(global_feature_grid)

        # The transformation is in the local coordinate system of the normalized grid [-1,1]^3, therefore scale the translation of the 
        # affine transforation to the world coordinate system 
        grid_size_scale_factor = 0.5*(self.global_voxel_dim-1)*self.global_voxel_inc 
        grid_transformation[:, :3, 3] = grid_size_scale_factor*grid_transformation[:, :3, 3]
        transformed_grid_origin = torch.bmm(global_Rot.squeeze(1), grid_transformation[:, :3, 3].unsqueeze(-1)).squeeze(-1) + self.global_origin.squeeze().repeat(bs,1).to(device)

        global_disp = global_disp.squeeze(1).view(bs, 3, -1)   # (B, 3, GH*GW*GD)
        # Undo the (random) global rotation of the grid
        global_disp = torch.bmm(global_Rot.squeeze(1).transpose(1,2), global_disp)   # (B, 3, GH*GW*GD)
        ones = torch.ones(bs, 1, global_disp.shape[-1]).to(device)
        global_disp_homogeneous = torch.cat((global_disp, ones), axis=1)  # (B, 4, GH*GW*GD)
        # Apply the transformation predicted by the spatial transformer
        warped_global_disp = torch.bmm(grid_transformation, global_disp_homogeneous) # (B, 3, GH*GW*GD)
        # Apply back the (random) global rotation
        warped_global_disp = torch.bmm(global_Rot.squeeze(1), warped_global_disp) # (B, 3, GH*GW*GD)
        # Translate warped grid to the global origin 
        warped_grid = warped_global_disp + self.global_origin.squeeze().repeat(bs,1).view(bs,3,1).to(device)    # (B, 3, GH*GW*GD)
        warped_grid = warped_grid.view(bs, 1, 3, grid_h, grid_w, grid_d)

        # Sample features at the refined grid positions
        global_feature_grid = self.vfs.forward(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, warped_grid)
        global_feature_grid = torch.squeeze(global_feature_grid, dim=1)

        # Reconstruct global cost volume
        global_cost_volume = self.global_net(global_feature_grid)  # (B*1, L, Dg, Dg, Dg)

        # Normalize volume
        global_cost_volume = self.normalize_volume(global_cost_volume)

        # Softmax point estimation
        pts_global = compute_expectation(global_cost_volume, warped_grid.repeat(1,ln,1,1,1,1))  # (B,L,3)

        return pts_global, global_cost_volume, warped_grid, transformed_grid_origin, grid_scales
