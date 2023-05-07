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

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

class EdgeLoss(nn.Module):
    def __init__(self, num_vertices, faces, vertex_masks=None, mask_weights=None, mesh_sampler=None):
        super().__init__()
        vertices_per_edge = EdgeLoss.get_vertices_per_edge(num_vertices, faces)
        # vertex_weights = EdgeLoss.get_vertex_weights(num_vertices, vertex_masks, mask_weights, mesh_sampler)
        vertex_weights = EdgeLoss.get_vertex_weights(num_vertices, faces, vertex_masks, mask_weights, mesh_sampler)
        self.edge_weights = (vertex_weights[vertices_per_edge[:, 0]] + vertex_weights[vertices_per_edge[:, 1]]) / 2.
        self.edges_for = lambda x: x[:, vertices_per_edge[:, 0], :] - x[:, vertices_per_edge[:, 1], :]

    def forward(self, vertices1, vertices2):
        """
        Given two meshes of the same topology, returns the relative edge differences.
        """

        batch_size = vertices1.shape[0]
        device = vertices1.device

        edge_weights = torch.from_numpy(self.edge_weights).to(vertices1.dtype)
        edge_weights = edge_weights[None, :, None].repeat(batch_size, 1, 1).to(device)

        edges1 = torch.multiply(edge_weights, self.edges_for(vertices1))
        edges2 = torch.multiply(edge_weights, self.edges_for(vertices2))
        return torch.nn.MSELoss()(edges1, edges2)

    @staticmethod
    def get_vert_connectivity(num_vertices, faces):
        """
        Returns a sparse matrix (of size #verts x #verts) where each nonzero
        element indicates a neighborhood relation. For example, if there is a
        nonzero element in position (15,12), that means vertex 15 is connected
        by an edge to vertex 12.
        Adapted from https://github.com/mattloper/opendr/
        """

        vpv = sp.csc_matrix((num_vertices,num_vertices))
        # for each column in the faces...
        for i in range(3):
            IS = faces[:,i]
            JS = faces[:,(i+1)%3]
            data = np.ones(len(IS))
            ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
            mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
            vpv = vpv + mtx + mtx.T
        return vpv

    @staticmethod
    def get_vertices_per_edge(num_vertices, faces):
        """
        Returns an Ex2 array of adjacencies between vertices, where
        each element in the array is a vertex index. Each edge is included
        only once. If output of get_faces_per_edge is provided, this is used to
        avoid call to get_vert_connectivity()
        Adapted from https://github.com/mattloper/opendr/
        """

        vc = sp.coo_matrix(EdgeLoss.get_vert_connectivity(num_vertices, faces))
        result = np.hstack((col(vc.row), col(vc.col)))
        result = result[result[:,0] < result[:,1]] # for uniqueness
        return result

    @staticmethod
    def get_vertex_weights(num_vertices, faces, vertex_masks=None, mask_weights=None, mesh_sampler=None):
        if vertex_masks is None or mask_weights is None:
            return np.ones(num_vertices)

        if (vertex_masks['vertex_count'] != num_vertices) and (mesh_sampler is None):
            raise RuntimeError("Mismatch of vertex counts with the loaded mask: %d != %d" % (num_vertices, vertex_masks['vertex_count']))

        if vertex_masks['vertex_count'] == num_vertices:
            verts_per_edge = EdgeLoss.get_vert_connectivity(num_vertices, faces)
        else:
            source_level = mesh_sampler.get_level(vertex_masks['vertex_count'])
            source_mesh = mesh_sampler.get_mesh(source_level)
            verts_per_edge = EdgeLoss.get_vert_connectivity(source_mesh.v.shape[0], source_mesh.f)

        def set_weights(vertex_weights, vertex_mask, vertex_weight, num_rings=0):
            vertex_weights[vertex_mask] = vertex_weight
            if num_rings > 0:
                extended_mask = EdgeLoss.get_extended_mask(verts_per_edge, vertex_mask, num_rings=num_rings, diff=True)
                vertex_weights[extended_mask] = vertex_weight/2

        vertex_weights = np.ones(vertex_masks['vertex_count'])
        set_weights(vertex_weights, vertex_masks['face'], mask_weights['w_edge_face'], num_rings=0)    
        set_weights(vertex_weights, vertex_masks['left_eyeball'], mask_weights['w_edge_eyeballs'], num_rings=0)
        set_weights(vertex_weights, vertex_masks['right_eyeball'], mask_weights['w_edge_eyeballs'], num_rings=0)
        set_weights(vertex_weights, vertex_masks['left_ear'], mask_weights['w_edge_ears'], num_rings=1)
        set_weights(vertex_weights, vertex_masks['right_ear'], mask_weights['w_edge_ears'], num_rings=1)
        set_weights(vertex_weights, vertex_masks['left_eye_region'], mask_weights['w_edge_eye_region'], num_rings=3)
        set_weights(vertex_weights, vertex_masks['right_eye_region'], mask_weights['w_edge_eye_region'], num_rings=3)
        set_weights(vertex_weights, vertex_masks['lips'], mask_weights['w_edge_lips'], num_rings=3)
        set_weights(vertex_weights, vertex_masks['neck'], mask_weights['w_edge_neck'], num_rings=0)
        set_weights(vertex_weights, vertex_masks['nostrils'], mask_weights['w_edge_nostrils'], num_rings=0)
        set_weights(vertex_weights, vertex_masks['scalp'], mask_weights['w_edge_scalp'], num_rings=0)
        set_weights(vertex_weights, vertex_masks['boundary'], mask_weights['w_edge_boundary'], num_rings=1)

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

    def get_extended_mask(verts_per_edge, mask_ids, num_rings=1, diff=True):
        '''
        Get mask of vertices extended by the k-ring neighborhood
        :params template        template mesh
        :params mask_ids        vertex ids of the mask
        :params num_ring        number of rings to be extended
        :params diff            if True, only the extended vertex ids are return
        return extended mask
        '''

        init_mask_ids = mask_ids.copy()
        for _ in range(num_rings):
            mask_ids = np.unique(verts_per_edge[mask_ids].nonzero()[1])
        
        if diff:
            return np.setdiff1d(mask_ids, init_mask_ids)
        else:
            return mask_ids


# -----------------------------------------------------------------------------

def test():
    from psbody.mesh import Mesh
    from mesh_sampling import MeshSampler

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    w_edge_face = 1.0
    w_edge_ears = 1.0
    w_edge_eyeballs = 1.0
    w_edge_eye_region = 1.0
    w_edge_lips = 1.0
    w_edge_neck = 1.0
    w_edge_nostrils = 1.0
    w_edge_scalp = 1.0
    w_edge_boundary = 1.0

    # mesh_sampling_list=[]
    mesh_sampling_list=[500,300,700]
    # mesh_sampling_list=[300,500]

    reg1_fname = './data/template/template_low_res_tri.obj'
    reg2_fname = './data/template/sampling_template.obj'

    vertex_masks = np.load('/is/ps3/tbolkart/misc_repo/ToFu_dev/data/template/vertex_masks.npz')
    mask_weights = {
        'w_edge_face': w_edge_face,
        'w_edge_ears': w_edge_ears,
        'w_edge_eyeballs': w_edge_eyeballs,
        'w_edge_eye_region': w_edge_eye_region,
        'w_edge_lips': w_edge_lips,
        'w_edge_neck': w_edge_neck,
        'w_edge_nostrils': w_edge_nostrils,
        'w_edge_scalp': w_edge_scalp,
        'w_edge_boundary': w_edge_boundary
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
        edge_loss_function = EdgeLoss(mesh1.v.shape[0], mesh1.f, vertex_masks=vertex_masks, mask_weights=mask_weights)
        vertices1 = torch.from_numpy(mesh1.v).to(torch.float32).unsqueeze(0).to(device)
        vertices2 = torch.from_numpy(mesh2.v).to(torch.float32).unsqueeze(0).to(device)
        edge_loss = edge_loss_function(vertices1, vertices2)
    else:
        # Providing mesh sampler, mesh at sampled resolution
        edge_loss_function = EdgeLoss(mesh1.v.shape[0], mesh1.f, vertex_masks=vertex_masks, mask_weights=mask_weights, mesh_sampler=mesh_sampler)
        vertices1 = torch.from_numpy(mesh1.v).to(torch.float32).unsqueeze(0).to(device)
        vertices2 = torch.from_numpy(mesh2.v).to(torch.float32).unsqueeze(0).to(device)
        edge_loss = edge_loss_function(vertices1, vertices2)

    import pdb; pdb.set_trace()

if __name__ == '__main__':
    test()