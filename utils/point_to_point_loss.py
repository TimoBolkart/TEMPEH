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
from torch import nn
import scipy.sparse as sp
import numpy as np

class PointToPointLoss(nn.Module):
    def __init__(self, num_vertices, vertex_masks=None, mask_weights=None, mesh_sampler=None, loss_function=torch.nn.MSELoss()):
        super().__init__()

        self.loss_function = loss_function
        self.vertex_weights = PointToPointLoss.get_vertex_weights(num_vertices, vertex_masks, mask_weights, mesh_sampler)

    def forward(self, vertices1, vertices2):
        """
        Given two meshes of the same topology, returns the relative edge differences.
        """

        batch_size = vertices1.shape[0]
        device = vertices1.device

        vertex_weights = torch.from_numpy(self.vertex_weights).to(vertices1.dtype)
        vertex_weights = vertex_weights[None, :, None].repeat(batch_size, 1, 1).to(device)

        vertices1 = torch.multiply(vertex_weights, vertices1)
        vertices2 = torch.multiply(vertex_weights, vertices2)
        return self.loss_function(vertices1, vertices2)

    @staticmethod
    def get_vertex_weights(num_vertices, vertex_masks=None, mask_weights=None, mesh_sampler=None):
        if vertex_masks is None or mask_weights is None:
            return np.ones(num_vertices)

        if (vertex_masks['vertex_count'] != num_vertices) and (mesh_sampler is None):
            raise RuntimeError("Mismatch of vertex counts with the loaded mask: %d != %d" % (num_vertices, vertex_masks['vertex_count']))

        vertex_weights = np.ones(vertex_masks['vertex_count'])
        if 'w_point_face' in mask_weights: vertex_weights[vertex_masks['face']] = mask_weights['w_point_face']
        if 'w_point_ears' in mask_weights: vertex_weights[vertex_masks['left_ear']] = mask_weights['w_point_ears']
        if 'w_point_ears' in mask_weights: vertex_weights[vertex_masks['right_ear']] = mask_weights['w_point_ears']
        if 'w_point_eyeballs' in mask_weights: vertex_weights[vertex_masks['left_eyeball']] = mask_weights['w_point_eyeballs']
        if 'w_point_eyeballs' in mask_weights: vertex_weights[vertex_masks['right_eyeball']] = mask_weights['w_point_eyeballs']
        if 'w_point_eye_region' in mask_weights: vertex_weights[vertex_masks['left_eye_region']] = mask_weights['w_point_eye_region']
        if 'w_point_eye_region' in mask_weights: vertex_weights[vertex_masks['right_eye_region']] = mask_weights['w_point_eye_region']
        if 'w_point_lips' in mask_weights: vertex_weights[vertex_masks['lips']] = mask_weights['w_point_lips']
        if 'w_point_neck' in mask_weights: vertex_weights[vertex_masks['neck']] = mask_weights['w_point_neck']
        if 'w_point_nostrils' in mask_weights: vertex_weights[vertex_masks['nostrils']] = mask_weights['w_point_nostrils']
        if 'w_point_scalp' in mask_weights: vertex_weights[vertex_masks['scalp']] = mask_weights['w_point_scalp']
        if 'w_point_boundary' in mask_weights: vertex_weights[vertex_masks['boundary']] = mask_weights['w_point_boundary']

        if vertex_masks['vertex_count'] != num_vertices:
            # Transfer vertex mask to the sampled mesh resolution
            source_level = mesh_sampler.get_level(vertex_masks['vertex_count'])
            target_level = mesh_sampler.get_level(num_vertices)

            num_levels = mesh_sampler.get_number_levels()
            for _ in range(num_levels):
                if source_level == target_level:
                    break
                vertex_weights = mesh_sampler.downsample(vertex_weights)
                source_level = mesh_sampler.get_level(vertex_weights.shape[0])
            vertex_weights = vertex_weights.reshape(-1,)
            if source_level != target_level:
                raise RuntimeError("Unable to downsample mesh to target level")
        return vertex_weights


# -----------------------------------------------------------------------------

def test():
    from psbody.mesh import Mesh
    from mesh_sampling import MeshSampler

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    w_point_face = 1.0
    w_point_ears = 1.0
    w_point_eyeballs = 1.0
    w_point_eye_region = 1.0
    w_point_lips = 1.0
    w_point_neck = 1.0
    w_point_nostrils = 1.0
    w_point_scalp = 1.0
    w_point_boundary = 1.0

    mesh_sampling_list=[]
    # mesh_sampling_list=[500,300,700]
    # mesh_sampling_list=[300,500]

    reg1_fname = './data/template/template_low_res_tri.obj'
    reg2_fname = './data/template/sampling_template.obj'

    vertex_masks = np.load('/is/ps3/tbolkart/misc_repo/ToFu_dev/data/template/vertex_masks.npz')
    mask_weights = {
        'w_point_face': w_point_face,
        'w_point_ears': w_point_ears,
        'w_point_eyeballs': w_point_eyeballs,
        'w_point_eye_region': w_point_eye_region,
        'w_point_lips': w_point_lips,
        'w_point_neck': w_point_neck,
        'w_point_nostrils': w_point_nostrils,
        'w_point_scalp': w_point_scalp,
        'w_point_boundary': w_point_boundary
    }

    mesh1 = Mesh(filename=reg1_fname)
    mesh2 = Mesh(filename=reg2_fname)
    
    if len(mesh_sampling_list) > 0:
        template_fname = './data/template/sampling_template.obj'    
        template_mesh = Mesh(filename=template_fname)
        mesh_sampler = MeshSampler(template_mesh, mesh_dimension_list=[500], keep_boundary_adjacent=True)

        for _ in range(len(mesh_sampling_list)):
            v1, f1 = mesh_sampler.downsample(mesh1.v, return_faces=True)
            mesh1 = Mesh(v1, f1)
            v2, f2 = mesh_sampler.downsample(mesh2.v, return_faces=True)
            mesh2 = Mesh(v2, f2)

    if len(mesh_sampling_list) == 0:
        # No mesh sampler, mesh at full resolution
        loss_function = PointToPointLoss(mesh1.v.shape[0])
        # loss_function = PointToPointLoss(mesh1.v.shape[0], vertex_masks=vertex_masks, mask_weights=mask_weights)
        vertices1 = torch.from_numpy(mesh1.v).to(torch.float32).unsqueeze(0).to(device)
        vertices2 = torch.from_numpy(mesh2.v).to(torch.float32).unsqueeze(0).to(device)
        p2p_loss = loss_function(vertices1, vertices2)
    else:
        # Providing mesh sampler, mesh at sampled resolution
        loss_function = PointToPointLoss(mesh1.v.shape[0], vertex_masks=vertex_masks, mask_weights=mask_weights, mesh_sampler=mesh_sampler)
        vertices1 = torch.from_numpy(mesh1.v).to(torch.float32).unsqueeze(0).to(device)
        vertices2 = torch.from_numpy(mesh2.v).to(torch.float32).unsqueeze(0).to(device)
        p2p_loss = loss_function(vertices1, vertices2)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    test()