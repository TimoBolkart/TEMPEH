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
from utils.mesh_helper import MeshHelper
from models.model_aligner.base_model import BaseModel
from modules.local_volumetric_feature_sampler import VolumetricFeatureSampler
from utils.positional_encoding import get_embedder

# -----------------------------------------------------------------------------

class Model(BaseModel):
    
    def __init__(self, 
                input_ch, 
                mesh_sampler,
                local_architecture='v2v',
                local_voxel_dim=16,
                local_voxel_inc_list=[2.0],
                global_embedding_type=None,
                norm='bn', 
                local_feature_fusion='mean_var',
                **kwargs):                
        super(Model, self).__init__()

        # ---- properties ----
        self.input_ch = input_ch  # input feature dim
        # self.number_points = number_points
        self.number_levels = mesh_sampler.get_number_levels()
        self.mesh_sampler = mesh_sampler

        # global voxel setting
        self.local_architecture = local_architecture
        self.local_voxel_dim = local_voxel_dim
        self.local_voxel_inc_list = local_voxel_inc_list
        self.norm = norm
        self.local_feature_fusion = local_feature_fusion
        self.normals_required = 'normal' in local_feature_fusion
        self.visibility_required = 'visibility' in local_feature_fusion

        # volumetric feature sampler
        self.fuse_ops_num = 2  # hardcoded, same as len(VolumetricFeatureSampler.fuse_ops)

        # global mesh embedding (included in MeshResampler)
        self.global_embedding_type = global_embedding_type
        if global_embedding_type in [None, '', 'none']:
            self.global_embedding_dim = 0
        elif global_embedding_type in ['coords']:
            self.global_embedding_dim = 3
        elif global_embedding_type[:2] == 'pe':
            from utils.positional_encoding import get_embedder
            num_pe_frequencies = int(global_embedding_type[2:])
            if num_pe_frequencies > 0:
                self.embedder, self.global_embedding_dim = get_embedder(num_pe_frequencies)
            else:
                self.global_embedding_dim = 3
        else:
            raise RuntimeError(f"Invalid global_embedding_type = {global_embedding_type}")\

        # ---- submodules ----
        self.module_names = ['local_net']

        if self.local_architecture == 'v2v':
            from modules.v2v_refinement import V2VModel
            self.local_net = V2VModel(
                input_channels=self.fuse_ops_num*len(self.local_voxel_inc_list)*self.input_ch + self.global_embedding_dim,
                output_channels=1,
                encdec_level=2,
                norm=self.norm)  
        else:
            raise RuntimeError( "unrecognizable global_architecture: %s" % ( self.local_architecture ) )

        self.vfs = VolumetricFeatureSampler(feature_fusion=self.local_feature_fusion)


    def print_setting(self):
        pass

    def normalize_volume(self, volume):
        ''' normalize volume spatially (throughout volume dimensions)
        Args and Returns:
            volume: (B, L, D, D, D)
        '''
        bs, ln, dim, dim, dim = volume.shape
        volume = torch.nn.Softmax(dim=2)(
            volume.view(bs, ln, -1)).view(bs, ln, dim, dim, dim)
        return volume

    def sample_local_features(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, points, 
                                    normals=None, visibilities=None, camera_centers=None, random_grid=False):
        '''sample volumetric features from local voxel grids
        Args:
            feature_maps: tensor in (B, V, F, H, W), 2d features from images
            camera_intrinsics:  (B, V, 3, 3), itrinsic transformation matrix
            camera_extrinsics:  (B, V, 3, 4), extrinsic transformation matrix
            camera_distortions: (B, V, 2), radial distortions
            points:             (B, L, 3), predicted coarse stage points used as center points for the refinement feature grids
            normals:            (B, L, 3), vertex normals for each of the coarse stage points
            visibilities:       (B, V, L), vertex visibility for each of the coarse stage points
            camera_centers:     (B, V, 3), camera centers
            random_grid:        Flag to indicate if the local grids are randomly rotated as a form of data augmentation
        Returns:
            local_feat_grids: (B, L, F', D, D, D), F' = F * len(self.local_voxel_inc_list) * len(self.mv_fuse_ops)
            local_grids: list of (B, L, 3, D, D, D)
            local_disps: list of (B, L, 3, D, D, D)
            local_Rot: (B, L, 3, 3)
        '''

        bs, vn, fd, fh, fw = feature_maps.shape
        device = feature_maps.device
        vd = self.local_voxel_dim  # D
        num_points = points.shape[1]  # L, sparse init point number

        # create grid ((B,L,3,3), randomly rotated voxel grids)
        from modules.voxel_utils import sample_random_rotation, generate_local_grids
        local_Rot = sample_random_rotation(bs, num_points).to(device) if random_grid else None

        # sample at projected locations of local detector grids
        local_feat_grids, local_grids, local_disps = [], [], []
        for gid, grid_inc in enumerate(self.local_voxel_inc_list):
            # create grid
            this_grid, this_disp = generate_local_grids(
                vert=points,
                grid_dim=vd,
                grid_inc=grid_inc,
                rotate_mat=local_Rot
            )

            # sample volumetric features
            this_feat_grid = self.vfs.forward(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, this_grid, 
                                            normals=normals, visibilities=visibilities, camera_centers=camera_centers)

            local_feat_grids.append(this_feat_grid)
            local_grids.append(this_grid)
            local_disps.append(this_disp)

        # combine
        local_feat_grids = torch.cat(local_feat_grids, dim=2)  # (B, L, F', D, D, D)

        if self.global_embedding_dim > 0:
            level = self.mesh_sampler.get_level(num_points)
            global_embedding = torch.from_numpy(0.01*self.mesh_sampler.get_mesh(level).v).to(local_feat_grids.dtype)
            if self.global_embedding_dim > 3:
                global_embedding = self.embedder(global_embedding)
            global_embedding = global_embedding[None, :, :, None, None, None].repeat(bs, 1, 1, vd, vd, vd).to(device)
            local_feat_grids = torch.cat((local_feat_grids, global_embedding), dim=2) 

        return local_feat_grids, local_grids, local_disps, local_Rot

    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, camera_centers, global_points, random_grid=True):
        from modules.voxel_utils import compute_expectation
        bs, vn, fd, fh, fw = feature_maps.shape
        vd = self.local_voxel_dim

        def compute_vertex_offsets(points, normals=None, visibilities=None):
            num_points = points.shape[1]

            # Sample features at the point locations
            local_feat_grids, _, local_disps, _ = self.sample_local_features(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, points, 
                                                                            normals=normals, visibilities=visibilities, camera_centers=camera_centers, random_grid=random_grid)

            # Predict local refinement vector
            local_feat_grids = local_feat_grids.view(bs*num_points, -1, vd, vd, vd) # outputs (batch_size*num_points, feature_dim+embedding_dim, vol_size, vol_size, vol_size)
            local_cost_vol = self.local_net(local_feat_grids) # outputs (batch_size*num_points, 1, vol_size, vol_size, vol_size)

            # Normalize and reshape cost volume
            local_cost_vol = self.normalize_volume(local_cost_vol).view(bs, num_points, vd, vd, vd) # the "1"-dim is squeezed

            # Compute vertex offset vector
            vertex_offsets = compute_expectation(local_cost_vol, local_disps[-1]).view(bs, num_points, -1) # expectation computeted on the small scale
            return vertex_offsets
       
        # Compute vertex normals and visibility map
        if self.normals_required or self.visibility_required:
            level = self.mesh_sampler.get_level(global_points.shape[1])
            mesh_helper = MeshHelper(global_points.shape[1], self.mesh_sampler.get_mesh(level).f)  
        
        vertex_normals, vertex_visibility = None, None
        if self.normals_required:
            vertex_normals = mesh_helper.vertex_normals(global_points)
        if self.visibility_required:
            # vertex_visibility = mesh_helper.vertex_visibility(global_points, camera_centers)
            vertex_visibility = mesh_helper.depth_vertex_visibility(global_points, camera_intrinsics, camera_extrinsics, depth_rendering_size=200)

        points_lower = global_points
        points_list = []
        # Iteratively upsample and refine points
        for level in range(self.number_levels-2, -1, -1):
            # Upsample points
            points_higher = self.mesh_sampler.batch_upsample(points_lower, return_faces=False)

            if self.normals_required:
                vertex_normals = self.mesh_sampler.batch_upsample(vertex_normals, return_faces=False)
                vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)
            if self.visibility_required:
                vertex_visibility = vertex_visibility.transpose(1,2)    # (B, num_points, V)
                vertex_visibility = self.mesh_sampler.batch_upsample(vertex_visibility, return_faces=False)
                vertex_visibility = torch.nn.Sigmoid()(100*vertex_visibility)
                vertex_visibility = vertex_visibility.transpose(1,2)    # (B, V, num_points')

            vertex_offsets = compute_vertex_offsets(points_higher, vertex_normals, vertex_visibility)
            points_higher = points_higher + vertex_offsets

            points_list.append(points_higher)
            points_lower = points_higher
        return points_list        
