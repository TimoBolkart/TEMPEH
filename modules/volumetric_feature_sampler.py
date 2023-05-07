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

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class VolumetricFeatureSampler(nn.Module):

    def __init__(self, padding_mode='zeros', feature_fusion='', feature_dim=0):
        super(VolumetricFeatureSampler, self).__init__()

        """ define a volumetric feature sampler
        """
        self.padding_mode = padding_mode

        # note: the two operations will cumulate the feature and squared feature, independently among the feature channels
        # later in self.compute_stats(), we convert the results into means and standard deviations.
        # currently these are hardcoded operations, and all tensors in this class with some dimension "2F" are also due to this reason
        self.fuse_ops = [
            lambda feat, base: feat if base is None else feat + base,
            lambda feat, base: feat.pow(2.0) if base is None else feat.pow(2.0) + base,
        ]
        self.fuse_ops_num = len(self.fuse_ops)

        if feature_fusion in [None, '', 'baseline_mean_var']:
            self.compute_feature_volume = self.baseline_mean_var        
        elif feature_fusion == 'mean_var':
            self.compute_feature_volume = self.mean_var
        elif feature_fusion == 'soft_weighted_mean_var':
            self.compute_feature_volume = self.weighted_mean_var
            self.soft_weighting = True
        elif feature_fusion == 'weighted_mean_var':
            self.compute_feature_volume = self.weighted_mean_var   
            self.soft_weighting = False         
        elif feature_fusion == 'filtered_mean_var':
            self.compute_feature_volume = self.filtered_mean_var   
        elif feature_fusion == 'filtered_soft_weighted_mean_var':
            self.compute_feature_volume = self.filtered_weighted_mean_var                   
            self.soft_weighting = True
        elif feature_fusion == 'filtered_weighted_mean_var':
            self.compute_feature_volume = self.filtered_weighted_mean_var                    
            self.soft_weighting = False
        elif feature_fusion == 'cov_soft_weighted_mean_var':
            self.compute_feature_volume = self.cov_weighted_mean_var
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.2))
            self.soft_weighting = True
        elif feature_fusion == 'cov_weighted_mean_var':
            self.compute_feature_volume = self.cov_weighted_mean_var
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.2))
            self.soft_weighting = False                        
        elif feature_fusion == 'similarity_soft_weighted_mean_var':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer
            self.fsl = FeatureSimilarityLayer(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = True
        elif feature_fusion == 'similarity_weighted_mean_var':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer
            self.fsl = FeatureSimilarityLayer(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = False      
        elif feature_fusion == 'similarity_soft_weighted_mean_var_v2':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer2
            self.fsl = FeatureSimilarityLayer2(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = True
        elif feature_fusion == 'similarity_weighted_mean_var_v2':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer2
            self.fsl = FeatureSimilarityLayer2(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = False  
        elif feature_fusion == 'similarity_soft_weighted_mean_var_v3':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer3
            self.fsl = FeatureSimilarityLayer3(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = True
        elif feature_fusion == 'similarity_weighted_mean_var_v3':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer3
            self.fsl = FeatureSimilarityLayer3(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = False      
        elif feature_fusion == 'similarity_soft_weighted_mean_var_v4':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer4
            self.fsl = FeatureSimilarityLayer4(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = True
        elif feature_fusion == 'similarity_weighted_mean_var_v4':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer4
            self.fsl = FeatureSimilarityLayer4(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = False     
        elif feature_fusion == 'similarity_soft_weighted_mean_var_v5':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer5
            self.fsl = FeatureSimilarityLayer5(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = True
        elif feature_fusion == 'similarity_weighted_mean_var_v5':
            self.compute_feature_volume = self.similarity_weighted_mean_var
            from modules.feature_similarity_layer import FeatureSimilarityLayer5
            self.fsl = FeatureSimilarityLayer5(in_features=feature_dim)
            self.feature_similarity_threshold = nn.Parameter(torch.tensor(0.0))
            self.soft_weighting = False     
        elif feature_fusion == 'learned_feature_fusion':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureFusionModule
            self.feature_fusion = FeatureFusionModule(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim)
        elif feature_fusion == 'learned_feature_fusion2':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureFusionModule2
            self.feature_fusion = FeatureFusionModule2(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim)            
        elif feature_fusion == 'learned_feature_similarity_fusion':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusionModule
            self.feature_fusion = FeatureSimilarityFusionModule(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim)
        elif feature_fusion == 'learned_soft_feature_similarity_fusion':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusionModule
            self.feature_fusion = FeatureSimilarityFusionModule(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim, soft_weighting=True)            
        elif feature_fusion == 'learned_feature_similarity_fusion2':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusionModule2
            self.feature_fusion = FeatureSimilarityFusionModule2(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim)
        elif feature_fusion == 'learned_soft_feature_similarity_fusion2':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusionModule2
            self.feature_fusion = FeatureSimilarityFusionModule2(in_features=feature_dim, out_features=self.fuse_ops_num*feature_dim, soft_weighting=True)                 
        
        
        elif feature_fusion == 'feature_similarity_fusion1':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion1
            self.feature_fusion = FeatureSimilarityFusion1(in_features=feature_dim)        
        elif feature_fusion == 'feature_similarity_fusion2':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion2
            self.feature_fusion = FeatureSimilarityFusion2(in_features=feature_dim)       
        elif feature_fusion == 'feature_similarity_fusion3':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion3
            self.feature_fusion = FeatureSimilarityFusion3(in_features=feature_dim)       
        elif feature_fusion == 'feature_similarity_fusion4':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion4
            self.feature_fusion = FeatureSimilarityFusion4(in_features=feature_dim)       
        elif feature_fusion == 'feature_similarity_fusion5':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion5
            self.feature_fusion = FeatureSimilarityFusion5(in_features=feature_dim)       
        elif feature_fusion == 'feature_similarity_fusion6':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion6
            self.feature_fusion = FeatureSimilarityFusion6(in_features=feature_dim)                                                               
        elif feature_fusion == 'feature_similarity_fusion7':
            self.compute_feature_volume = self.learned_feature_fusion
            from modules.feature_similarity_layer import FeatureSimilarityFusion7
            self.feature_fusion = FeatureSimilarityFusion7(in_features=feature_dim)                
        else:
            raise RuntimeError( "Unrecognizable feature_fusion: %s" % ( feature_fusion ) )
                
    @staticmethod
    def project(points, camera_intrinsics, camera_extrinsics, camera_distortions, height, width, eps=1e-7):
        """ project the grid points into images (full perspective projection)
        Args:
        - points: (B, N, 3)
        - camera_intrinsics: (B, 3, 4)
        - camera_extrinsics: (B, 3, 3)
        - camera_distortions: (B, 2)
        - height: int 
        - width: int

        Return:
        - u_coord: (B, N), value range in [-1, 1]
        - v_coord: (B, N), value range in [-1, 1]
        """

        device = points.device
        batch_size, num_points, _ = points.shape
        points = points.transpose(1, 2).contiguous()

        ones = torch.ones(batch_size, 1, num_points).to(device)
        points_homogeneous = torch.cat((points, ones), axis=-2) # (batch_size, 4, num_points)

        # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
        points_image = camera_extrinsics.bmm(points_homogeneous) # (batch_size, 3, num_points)

        # Transformation from 3D camera coordinate system to the undistorted image plane 
        mask = (points_image.abs() < eps)
        mask[:,:2,:] = False
        points_image[mask] = 1.0 # Avoid division by zero

        points_image_x = points_image[:,0,:] / points_image[:,2,:]
        points_image_y = points_image[:,1,:] / points_image[:,2,:]

        # Transformation from undistorted image plane to distorted image coordinates
        K1, K2 = camera_distortions[:,0], camera_distortions[:,1]       # (batch_size)
        r2 = points_image_x**2 + points_image_y**2            # (batch_size, num_points)
        r4 = r2**2
        radial_distortion_factor = (1 + K1[:, None]*r2 + K2[:, None]*r4)  # (batch_size, num_points)

        points_image_x = points_image_x*radial_distortion_factor
        points_image_y = points_image_y*radial_distortion_factor
        points_image_z = torch.ones_like(points_image[:,2,:])
        points_image = torch.cat((points_image_x[:, None, :], points_image_y[:, None, :], points_image_z[:, None, :]), dim=1)

        # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
        points_image = camera_intrinsics.bmm(points_image)      # (batch_size, 3, num_points)

        # Convert the range to [-1, 1]
        u_coord = 2. * points_image[:, 0, :] / (width - 1.) - 1.
        v_coord = 2. * points_image[:, 1, :] / (height - 1.) - 1.
        return u_coord, v_coord

    @staticmethod
    def compute_stats(feat_grid, view_num, split_dim=2):
        """ convert the feature grid from 1st- and 2nd-order moments to actual average and std
        Args:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        - view_num: int
        - split_dim: int, which dim to split, currently should always set to 2 (corresp. to the dim of "2F")

        Return:
        - feat_grid: (B, G, 2F, GH, GW, GD)
        """
        feat_grid /= float(view_num)
        feat_mean, feat_var = feat_grid.chunk(2, dim=split_dim)
        feat_var = feat_var - feat_mean.pow(2)
        feat_grid = torch.cat((feat_mean, feat_var), dim=split_dim)
        return feat_grid

    def baseline_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        voxels_all = []
        for opid, op in enumerate(self.fuse_ops):  # operation
            voxel = None

            for view_idx in range(num_views):
                # projection
                u_coord, v_coord = self.project(xyzs, camera_intrinsics[:, view_idx], camera_extrinsics[:, view_idx], camera_distortions[:, view_idx], height, width)  # u_coord.shape = (1, 32768)

                # sample
                grid2d_uv = torch.stack((u_coord, v_coord), dim=2).view(batch_size, grid_num, -1, 2)
                feat2d_uv = F.grid_sample(feature_maps[:, view_idx], grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B, F, G, GH*GW*GD)
                feat2d_uv = feat2d_uv.view(batch_size, -1, grid_num, grid_h, grid_w, grid_d)  # (B, F, G, GH, GW, GD)

                if voxel is None:
                    voxel = op(feat2d_uv, None)
                else:
                    voxel = op(feat2d_uv, voxel)

            voxels_all.append(voxel.transpose(1, 2))  # (B, G, F, GH, GW, GD)

        # convert to mean and std
        voxels = torch.cat(voxels_all, dim=2).contiguous().view(batch_size, grid_num, -1, grid_h, grid_w, grid_d)  # (B, G, F*len(ops), GH, GW, GD)
        voxels = self.compute_stats(voxels, view_num=num_views)  # (B, G, 2F, GH, GW, GD)
        return voxels

    def mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2)    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        # voxels = torch.cat((feat2d_uv.mean(dim=1), feat2d_uv.std(dim=1, unbiased=False)), dim=2).contiguous()   # (B, G, 2F, GH, GW, GD)
        feature_volume = torch.cat((feat2d_uv.mean(dim=1), feat2d_uv.var(dim=1, unbiased=True)), dim=2).contiguous()   # (B, G, 2F, GH, GW, GD)
        return feature_volume

    def weighted_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):      
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)
        feat2d_uv = feat2d_uv.permute(0,2,4,5,6,1,3).contiguous()   # (B, G, GH, GW, GD, V, F)
        feat2d_uv = feat2d_uv.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
   
        feature_median, _ = feat2d_uv.median(dim=1)         # (B*G*GH*GW*GD, F)
        feature_median = feature_median.detach()
        feature_median = feature_median.view(-1, fd, 1)    # (B*G*GH*GW*GD, F, 1)
        feature_weights = torch.bmm(torch.nn.functional.normalize(feat2d_uv, dim=2), torch.nn.functional.normalize(feature_median, dim=1))  # (B*G*GH*GW*GD, V, 1)

        if self.soft_weighting:
            feature_weights = torch.nn.Softmax(dim=1)(10*feature_weights)
        else:
            feature_weights = torch.nn.ReLU()(feature_weights)  # (B*G*GH*GW*GD, V, 1)
            feature_weights = feature_weights/float(num_views)

        weighted_mean = torch.mul(feat2d_uv, feature_weights).sum(dim=1)   # (B*G*GH*GW*GD, F)
        weighted_var = torch.mul(feat2d_uv.pow(2), feature_weights).sum(dim=1)-weighted_mean.pow(2) # (B*G*GH*GW*GD, F)
    
        feature_volume = torch.cat((weighted_mean, weighted_var), dim=1).contiguous()        # (B*G*GH*GW*GD, 2F)
        feature_volume = feature_volume.view(batch_size, grid_num, grid_h, grid_w, grid_d, 2*fd)      # (B, G, GH, GW, GD, 2F)
        feature_volume = feature_volume.permute(0,1,5,2,3,4).contiguous()     # (B, G, 2F, GH, GW, GD)
        return feature_volume

    def filtered_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  # (B*V, G*GH*GW*GD)
        
        # mask for points that project inside the images
        inside_mask = torch.logical_and(torch.logical_and(u_coord>=-1.0, u_coord<=1.0), torch.logical_and(v_coord>=-1.0, v_coord<=1.0)).contiguous()    # (B*V, G*GH*GW*GD)
        inside_mask = inside_mask.detach()
        inside_mask = inside_mask.view(batch_size, num_views, grid_num, grid_h, grid_w, grid_d)     # (B, V, G, GH, GW, GD)
        inside_mask = inside_mask.unsqueeze(dim=3)  # (B, V, G, 1, GH, GW, GD)

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        factor = torch.clamp(inside_mask.sum(dim=1), min=1) # (B, G, 1, GH, GW, GD)
        mean = (feat2d_uv*inside_mask).sum(dim=1)/factor        # (B, G, F, GH, GW, GD)
        var = (feat2d_uv.pow(2)*inside_mask).sum(dim=1)/factor - mean.pow(2)    # (B, G, F, GH, GW, GD)
        feature_volume = torch.cat((mean, var), dim=2).contiguous()        # (B, G, 2F, GH, GW, GD)
        return feature_volume

    def filtered_weighted_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):      
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  # (B*V, G*GH*GW*GD)
        
        # mask for points that project ouside the images
        outside_mask = torch.logical_or(torch.logical_or(u_coord < -1.0, u_coord > 1.0), torch.logical_or(v_coord < -1.0, v_coord > 1.0))
        outside_mask = outside_mask.detach()
        outside_mask = outside_mask.view(batch_size, num_views, grid_num, grid_h, grid_w, grid_d)     # (B, V, G, GH, GW, GD)

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        # feature median cross views with points filtered out that map outside of the view images
        feat2d_uv_nan = feat2d_uv.masked_fill(outside_mask.unsqueeze(dim=3), float('nan')).detach()
        feat2d_uv_nan = feat2d_uv_nan.permute(0,2,4,5,6,1,3).contiguous()   # (B, G, GH, GW, GD, V, F)
        feat2d_uv_nan = feat2d_uv_nan.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
           
        feature_median, _ = feat2d_uv_nan.nanmedian(dim=1)         # (B*G*GH*GW*GD, F)
        # catch the points that are not projected into any of the view images and produce nan median vectors
        feature_median = torch.nan_to_num(feature_median, nan=0.0)
        feature_median = feature_median.view(-1, fd, 1)    # (B*G*GH*GW*GD, F, 1)

        # feature similarity weight as the cosine similarity of each feature vector and the median feature vector
        feat2d_uv = feat2d_uv.permute(0,2,4,5,6,1,3).contiguous()   # (B, G, GH, GW, GD, V, F)
        feat2d_uv = feat2d_uv.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
        feature_similarity_weights = torch.bmm(torch.nn.functional.normalize(feat2d_uv, dim=2), torch.nn.functional.normalize(feature_median, dim=1))  # (B*G*GH*GW*GD, V, 1)
        feature_similarity_weights = feature_similarity_weights.view(batch_size, grid_num, grid_h, grid_w, grid_d, num_views)   # (B, G, GH, GW, GD, V)
        feature_similarity_weights = feature_similarity_weights.permute(0,5,1,2,3,4).contiguous()   # (B, V, G, GH, GW, GD)

        # feature visibility weight (1 if a point projects inside an image, and 0 if outside)
        feature_visibility_weights = torch.clamp(1.0 - outside_mask.float(), min=0.0, max=1.0)     # (B, V, G, GH, GW, GD)

        # weight of each feature as the product of similarity and visibility, clamped to [0,1], where 0 is invisible / dissimilar, and 1 is visible & similar
        feature_weights = torch.clamp(torch.mul(feature_similarity_weights, feature_visibility_weights), min=0)     # (B, V, G, GH, GW, GD)

        if self.soft_weighting:
            feature_weights = torch.nn.Softmax(dim=1)(10*feature_weights)
            num_visibile_views = 1.0
        else:
            num_visibile_views = torch.clamp((1.0-outside_mask.float()).sum(dim=1), min=1) # (B, G, GH, GW, GD)
            num_visibile_views = num_visibile_views.unsqueeze(2)    # (B, G, 1, GH, GW, GD)

        feature_weights = feature_weights.unsqueeze(3)  # (B, V, G, 1, GH, GW, GD)
        feat2d_uv = feat2d_uv.view(batch_size, grid_num, grid_h, grid_w, grid_d, num_views, fd)  # (B, G, GH, GW, GD, V, F)
        feat2d_uv = feat2d_uv.permute(0,5,1,6,2,3,4).contiguous()  # (B, V, G, F, GH, GW, GD)

        weighted_mean = torch.mul(feat2d_uv, feature_weights).sum(dim=1)/num_visibile_views   # (B, G, F, GH, GW, GD)
        weighted_var = torch.mul(feat2d_uv.pow(2), feature_weights).sum(dim=1)/num_visibile_views-weighted_mean.pow(2) # (B, G, F, GH, GW, GD)
        feature_volume = torch.cat((weighted_mean, weighted_var), dim=2).contiguous()        # (B, G, 2F, GH, GW, GD)
        return feature_volume

    def cov_weighted_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):       
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  # (B*V, G*GH*GW*GD)
        
        # sample feature maps
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        # constrain similarity threshold to [-1,1]
        feature_similarity_threshold = torch.nn.Tanh()(self.feature_similarity_threshold)

        # compute feature similarity between the views
        view_features = feat2d_uv.permute(0,2,4,5,6,1,3).contiguous().detach()   # (B, G, GH, GW, GD, V, F)
        view_features = view_features.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
        view_features = torch.nn.functional.normalize(view_features, dim=-1)    # (B*G*GH*GW*GD, V, F)
        # feature similarity s_ij = <f_i, f_j> \in [-1, 1]
        feature_similarity = torch.bmm(view_features, view_features.transpose(1,2))  # (B*G*GH*GW*GD, V, V)
        # discard all dissimilar values (smaller than a similarity threshold) 
        feature_similarity = torch.nn.ReLU()(feature_similarity-feature_similarity_threshold)
        # compute the similarity scores by aggregating the similarities across all views
        similarity_scores = feature_similarity.sum(-1)   # (B*G*GH*GW*GD, V)
        max_score_views = torch.argmax(similarity_scores, dim=-1)  # (B*G*GH*GW*GD)

        view_similarity = feature_similarity[torch.arange(max_score_views.shape[0]), max_score_views, :]    # (B*G*GH*GW*GD, V)
        view_similarity = view_similarity-feature_similarity_threshold # (B*G*GH*GW*GD, V)
        feature_weights = torch.nn.Sigmoid()(100*view_similarity).contiguous()  # (B*G*GH*GW*GD, V)
        feature_weights = feature_weights.view(batch_size, grid_num, grid_h, grid_w, grid_d, num_views) # (B, G, GH, GW, GD, V)
        feature_weights = feature_weights.permute(0,5,1,2,3,4)  # (B, V, G, GH, GW, GD)

        if self.soft_weighting:
            feature_weights = torch.nn.Softmax(dim=1)(10*feature_weights)  # (B, V, G, GH, GW, GD)
            view_normalization = 1.0
        else:
            view_normalization = torch.clamp(feature_weights.sum(1), min=1.0) # (B, G, GH, GW, GD)
            view_normalization = view_normalization.unsqueeze(2)    # (B, G, 1, GH, GW, GD)
        feature_weights = feature_weights.unsqueeze(3)  # (B, V, G, 1, GH, GW, GD)

        weighted_mean = torch.mul(feat2d_uv, feature_weights).sum(dim=1)/view_normalization   # (B, G, F, GH, GW, GD)
        weighted_var = torch.mul(feat2d_uv.pow(2), feature_weights).sum(dim=1)/view_normalization-weighted_mean.pow(2) # (B, G, F, GH, GW, GD)
        feature_volume = torch.cat((weighted_mean, weighted_var), dim=2).contiguous()        # (B, G, 2F, GH, GW, GD)
        return feature_volume    
    
    def similarity_weighted_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  # (B*V, G*GH*GW*GD)
        
        # sample feature maps
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        # compute feature similarity between the views
        view_features = feat2d_uv.permute(0,2,4,5,6,1,3).contiguous()   # (B, G, GH, GW, GD, V, F)
        view_features = view_features.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
        feature_weights = self.fsl(view_features) # (B*G*GH*GW*GD, V)
        feature_weights = feature_weights.view(batch_size, grid_num, grid_h, grid_w, grid_d, num_views) # (B, G, GH, GW, GD, V)
        feature_weights = feature_weights.permute(0,5,1,2,3,4)  # (B, V, G, GH, GW, GD)

        # set similarities smaller than a threshold to 0, others to 1
        feature_similarity_threshold = torch.nn.Tanh()(self.feature_similarity_threshold)
        feature_weights = torch.nn.Sigmoid()(100*(feature_weights-feature_similarity_threshold)).contiguous()  # (B, V, G, GH, GW, GD)

        if self.soft_weighting:
            feature_weights = torch.nn.Softmax(dim=1)(feature_weights)     # (B, V, G, GH, GW, GD)
            view_normalization = 1.0
        else:
            view_normalization = torch.clamp(feature_weights.sum(1), min=1.0) # (B, G, GH, GW, GD)
            view_normalization = view_normalization.unsqueeze(2)    # (B, G, 1, GH, GW, GD)
        feature_weights = feature_weights.unsqueeze(3)  # (B, V, G, 1, GH, GW, GD)

        weighted_mean = torch.mul(feat2d_uv, feature_weights).sum(dim=1)/view_normalization   # (B, G, F, GH, GW, GD)
        weighted_var = torch.mul(feat2d_uv.pow(2), feature_weights).sum(dim=1)/view_normalization-weighted_mean.pow(2) # (B, G, F, GH, GW, GD)
        feature_volume = torch.cat((weighted_mean, weighted_var), dim=2).contiguous()        # (B, G, 2F, GH, GW, GD)
        return feature_volume   

    def learned_feature_fusion(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)               # (B, 3, G*GH*GW*GD)
        xyzs = xyzs.transpose(1, 2).contiguous() # (B, G*GH*GW*GD, 3)

        # projection
        xyzs = xyzs.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(xyzs, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  # (B*V, G*GH*GW*GD)
        
        # sample feature maps
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G*GH*GW*GD, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, grid_num, -1, 2) # (B*V, G, GH*GW*GD, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, GH*GW*GD)
        feat2d_uv = feat2d_uv.transpose(1, 2).contiguous()    # (B*V, G, F, GH*GW*GD)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, grid_num, -1, grid_h, grid_w, grid_d)  # (B, V, G, F, GH, GW, GD)

        # fuse features across views
        view_features = feat2d_uv.permute(0,2,4,5,6,1,3).contiguous()   # (B, G, GH, GW, GD, V, F)
        view_features = view_features.view(-1, num_views, fd)    # (B*G*GH*GW*GD, V, F)
        feature_volume = self.feature_fusion(view_features).contiguous() # (B*G*GH*GW*GD, 2F)
        feature_volume = feature_volume.view(batch_size, grid_num, grid_h, grid_w, grid_d, -1)  # (B, G, GH, GW, GD, 2F)
        feature_volume = feature_volume.permute(0,1,5,2,3,4)   # (B, G, 2F, GH, GW, GD)
        return feature_volume

###########
    def debug_feature_sampling(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, points):
        batch_size, num_views, fd, height, width = feature_maps.shape
        num_points = points.shape[1]

        # projection
        points = points.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(points, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, num_points, -1, 2) # (B*V, G, 1, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, 1)
        feat2d_uv = feat2d_uv.transpose(1, 2)    # (B*V, G, F, 1)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, num_points, -1)  # (B, V, G, F)
        return feat2d_uv

    def debug_predicted_visibility(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, points):
        batch_size, num_views, fd, height, width = feature_maps.shape
        num_points = points.shape[1]

        # projection
        points = points.unsqueeze(1).repeat(1,num_views,1,1).view(batch_size*num_views, -1, 3)
        u_coord, v_coord = self.project(points, camera_intrinsics.view(-1,3,3), camera_extrinsics.view(-1,3,4), camera_distortions.view(-1,2), height, width)  

        # sample
        grid2d_uv = torch.stack((u_coord, v_coord), dim=2) # (B*V, G, 2)
        grid2d_uv = grid2d_uv.view(batch_size*num_views, num_points, -1, 2) # (B*V, G, 1, 2)
        feat2d_uv = F.grid_sample(feature_maps.view(-1, fd, height, width), grid2d_uv, padding_mode=self.padding_mode, align_corners=False)  # (B*V, F, G, 1)
        feat2d_uv = feat2d_uv.transpose(1,2)    # (B*V, G, F, 1)
        feat2d_uv = feat2d_uv.view(batch_size, num_views, num_points, -1)  # (B, V, G, F)
        feat2d_uv = feat2d_uv.transpose(1,2).contiguous()     # (B, G, V, F)
        feat2d_uv = feat2d_uv.view(-1, num_views, fd)    # (B*G, V, F)
        feature_weights = self.feature_fusion.debug_feature_similarity(feat2d_uv).contiguous() # (B*G, V)
        feature_weights = feature_weights.view(batch_size, num_points, num_views) # (B, G, V)
        feature_weights = feature_weights.transpose(1,2)    # (B, V, G)
        return feature_weights
###########

    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids):
        """ the full forward process
        Args:
        - feature_maps: tensor in (B, V, F, H, W), batched feature map
        - camera_intrinsics: tensor in (B, V, 3, 3), batched intrisic matrices
        - camera_extrinsics: tensor in (B, V, 3, 4), batched rigid transformation (camera poses)
        - camera_distortions: tensor in (B, V, 2), batched radial camera distortions
        - grids: (B, G, 3, GH, GW, GD), batched grid coordinates

        B: batch size; F: feature dim; V: view num;
        G: grid num (global stage = 1, local stage = input vertex number)
        GH, GW, GD: shape of the grid, currently all equal to grid dim D

        Return:
        - voxels: tensor in (B, G, 2F, GH, GW, GD), sampled volumetric features
        """

        feature_volume = self.compute_feature_volume(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids)
        return feature_volume

