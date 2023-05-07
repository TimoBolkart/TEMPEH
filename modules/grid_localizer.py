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

from platform import java_ver
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import rotation_6d_to_matrix

class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='batch_norm'):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_channels) if norm == 'batch_norm' else nn.InstanceNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Res3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='batch_norm'):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_channels) if norm == 'batch_norm' else nn.InstanceNorm3d(out_channels),
            nn.ReLU(True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_channels) if norm == 'batch_norm' else nn.InstanceNorm3d(out_channels)
        )

        if in_channels == out_channels:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_channels) if norm == 'batch_norm' else nn.InstanceNorm3d(out_channels)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)
   

class RigidGridLocalizer(nn.Module):
    def __init__(self, input_channels, volume_dim, hidden_channels=32, kernel_size=3, num_levels=2, regress_rotation=True, scale_factor=1.0, pooling=True, norm='batch_norm', dropout=0.0):
        super().__init__()

        self.regress_rotation = regress_rotation
        self.scale_factor = scale_factor

        # Spatial transformer localization-network
        tmp_in_channels = input_channels
        localization_layers = []
        for _ in range(num_levels):
            localization_layers.append(nn.Conv3d(in_channels=tmp_in_channels, out_channels=hidden_channels, 
                                                kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)))
            if norm == 'batch_norm':
                localization_layers.append(nn.BatchNorm3d(hidden_channels))
            if pooling:
                localization_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            localization_layers.append(nn.ReLU(True))
            tmp_in_channels = hidden_channels
        self.localization = nn.Sequential(*localization_layers)

        if pooling:
            downsampling_factor = 1./float(2**num_levels)
            pooled_volume_dim = max(int(downsampling_factor*volume_dim), 1)
        else:
            pooled_volume_dim = volume_dim
        in_features = hidden_channels*pow(pooled_volume_dim, 3)

        num_out_features = 3 + 3    # unisotropic scale + 3D translation
        if regress_rotation:
            num_out_features += 6   # rotation in 6D representation

        # Regressor for the transformation matrices
        mapping_layers = []
        mapping_layers.append(nn.Linear(in_features=in_features, out_features=3*hidden_channels, bias=True))
        if dropout:
            mapping_layers.append(nn.Dropout(dropout))
        mapping_layers.append(nn.ReLU(True))
        mapping_layers.append(nn.Linear(in_features=3*hidden_channels, out_features=num_out_features, bias=True))
        self.mapping = nn.Sequential(*mapping_layers)

        self._initialize_weights()
   
    def forward(self, feature_grid):
        '''
        Spatial transformer to regress transformation parameters from a feature grid

        Args:
            feature_grid: (B, F, GH, GW, GD)
        '''

        x = self.localization(feature_grid)
        x = x.view(x.shape[0], -1)
        x = self.mapping(x)
        # scales = self.scale_factor*torch.nn.Sigmoid()(x[:,:3])     # (B, 3)
        scales = self.scale_factor*0.5*(torch.nn.Tanh()(x[:,:3])+1.0)     # (B, 3)
        scales_mat = torch.diag_embed(scales, offset=0, dim1=-2, dim2=-1)
        if self.regress_rotation:
            rotation = rotation_6d_to_matrix(x[:,3:9])  # (B, 3, 3)
            scaled_rotation = torch.bmm(scales_mat, rotation)
            translation = torch.nn.Tanh()(x[:,9:])   # (B, 3)
        else:
            scaled_rotation = scales_mat
            translation = torch.nn.Tanh()(x[:,3:])   # (B, 3)  
        transformation = torch.cat((scaled_rotation, translation.unsqueeze(-1)), dim=-1)
        # import pdb; pdb.set_trace()
        return transformation, scales

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        # Initialize the weights/bias that the resulting transformation 
        # is close to (Tanh(3)=0.995.. for the scale) an identity transformation
        self.mapping[-1].weight.data.zero_()
        if self.regress_rotation:
            self.mapping[-1].bias.data.copy_(torch.tensor([ 3, 3, 3, 
                                                            1, 0, 0, 
                                                            0, 1, 0,
                                                            0, 0, 0], dtype=torch.float))                 
        else:               
            self.mapping[-1].bias.data.copy_(torch.tensor([ 3, 3, 3, 
                                                            0, 0, 0], dtype=torch.float))    


# -----------------------------------------------------------------------------

def generate_grid(batch_size, grid_dim, grid_inc):
    # single grid
    xx, yy, zz = torch.meshgrid([
        torch.arange( -grid_dim//2, grid_dim//2 ),
        torch.arange( -grid_dim//2, grid_dim//2 ),
        torch.arange( -grid_dim//2, grid_dim//2 )])

    # if grid_dim is odd, then make the grids "symmetrical" around zero
    if grid_dim % 2 == 1:
        xx, yy, zz = xx+1, yy+1, zz+1

    xx = xx.float() * float(grid_inc)
    yy = yy.float() * float(grid_inc)
    zz = zz.float() * float(grid_inc)
    disp_grid = torch.cat((xx[None,:], yy[None,:], zz[None,:]), dim=0) # (3,D,D,D)
    disp_grid = disp_grid[None,:,:,:,:].repeat(batch_size,1,1,1,1)
    return disp_grid