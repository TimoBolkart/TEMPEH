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

    def __init__(self, padding_mode='zeros', feature_fusion=''):
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

        if feature_fusion == 'mean_var':
            self.compute_feature_volume = self.mean_var
        elif 'visibility_filtered' in feature_fusion:
            self.compute_feature_volume = self.visibility_filtered_mean_var
        else:
            raise RuntimeError( "Unrecognizable feature_fusion: %s" % ( self.feature_fusion ) )
                
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

    def mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids, **kwargs):
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

    def visibility_filtered_mean_var(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids, visibilities, normals, camera_centers, **kwargself):
        '''
        Args:
            feature_maps: (B, V, F, H, W), image feature maps
            camera_intrinsics: (B, V, 3, 3)
            camera_extrinsics: (B, V, 3, 3)
            camera_distortions: (B, V, 3)
            grids: (B, G, 3, GH, GW, GD)
            visibilities: (B, V, G)
            normals: (B, G, 3)
            camera_centers: (B, V, 3)
        '''

        batch_size, num_views, fd, height, width = feature_maps.shape
        _, grid_num, _, grid_h, grid_w, grid_d = grids.shape
        xyzs = grids.transpose(1, 2).contiguous() # (B, 3, G, GH, GW, GD)
        xyzs = xyzs.view(batch_size, 3, -1)      # (B, 3, G*GH*GW*GD)
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

        # mask for points that project inside the images
        inside_mask = torch.logical_and(torch.logical_and(u_coord>=-1.0, u_coord<=1.0), torch.logical_and(v_coord>=-1.0, v_coord<=1.0)).contiguous()    # (B*V, G*GH*GW*GD)
        inside_mask = inside_mask.view(batch_size, num_views, grid_num, grid_h, grid_w, grid_d)     # (B, V, G, GH, GW, GD)

        # view_direction = grids - camera_centers
        view_directions = xyzs - camera_centers.contiguous().view(-1,1,3)     # (B*V, G*GH*GW*GD, 3)

        # cos_angle = <view_directions, normals>
        view_directions = view_directions.contiguous().view(batch_size, num_views, grid_num, -1, 3)     # (B, V, G, GH*GW*GD, 3)
        view_directions = view_directions.transpose(1,2).contiguous()  # (B, G, V, GH*GW*GD, 3)
        view_directions = view_directions.view(batch_size*grid_num, -1, 3)  # (B*G, V*GH*GW*GD, 3)
        view_directions = torch.nn.functional.normalize(view_directions, dim=-1)   # (B*G, V*GH*GW*GD, 3)
        view_normal_cos_angles = torch.bmm(view_directions, normals.contiguous().view(-1,3,1)) # (B*G, V*GH*GW*GD, 1)
        view_normal_cos_angles = view_normal_cos_angles.view(batch_size, grid_num, num_views, -1)   # (B, G, V, GH*GW*GD)
        view_normal_cos_angles = view_normal_cos_angles.transpose(1,2).contiguous()  # (B, V, G, GH*GW*GD)

        # Positive angles get zero weights
        view_normal_cos_angles[view_normal_cos_angles>0] = 0.0      # (B, V, G, GH*GW*GD)
        view_normal_cos_angles = view_normal_cos_angles.view((batch_size, num_views, grid_num, grid_h, grid_w, grid_d))   # (B, V, G, GH, GW, GD)

        # Weight negative cosine angles with visibility
        visibilities = visibilities.contiguous().view(batch_size, num_views, grid_num, 1, 1, 1)  # (B, V, G, 1, 1, 1)
        feature_weights = torch.multiply(visibilities, -view_normal_cos_angles)   # (B, V, G, GH, GW, GD)
        feature_weights = torch.multiply(feature_weights, inside_mask.float())   # (B, V, G, GH, GW, GD)

        # Make weights strictly positive and normalize them to sum to 1 across all views
        feature_weights = torch.nn.Softplus(beta=50, threshold=5)(feature_weights)  # (B, V, G, GH, GW, GD)
        feature_weights = feature_weights/feature_weights.sum(1).unsqueeze(1)   # (B, V, G, GH, GW, GD)
        feature_weights = feature_weights.unsqueeze(3)  # (B, V, G, 1, GH, GW, GD)

        # Computed weighted mean and standard deviation and concatenate them
        weighted_mean = torch.mul(feat2d_uv, feature_weights).sum(dim=1)   # (B, G, F, GH, GW, GD)
        weighted_var = torch.mul(feat2d_uv.pow(2), feature_weights).sum(dim=1)-weighted_mean.pow(2) # (B, G, F, GH, GW, GD)
        feature_volume = torch.cat((weighted_mean, weighted_var), dim=2).contiguous()        # (B, G, 2F, GH, GW, GD)
        return feature_volume


    def forward(self, feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids, normals=None, visibilities=None, camera_centers=None):
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

        feature_volume = self.compute_feature_volume(feature_maps, camera_intrinsics, camera_extrinsics, camera_distortions, grids, normals=normals, visibilities=visibilities, camera_centers=camera_centers)
        return feature_volume

