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
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

def compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces):
    mesh_face_vertices = index_vertices_by_faces(predicted_vertices, predicted_faces) # (batch_size, num_faces, num_vertices, 3)
    distances, _, _ = point_to_mesh_distance(scan_vertices, mesh_face_vertices)
    return distances.abs().sqrt().squeeze()

def GMO(sqr_input, sigma):
    '''
    Compute the square root of the Geman-McClure robustifier. As the input is assumed to be squared,
    the sign of the input is not considered. 
    '''
    eps = 1e-16
    sqr_sigma = sigma**2
    return torch.sqrt((sqr_sigma * (sqr_input / (sqr_sigma + sqr_input)))+eps)

class PointToSurfaceLoss(nn.Module):
    def __init__(self, gmo_sigma=1.0):
        super().__init__()
        
        if gmo_sigma <= 1e-8:
            self.robustifier = lambda x: x
        else:
            self.robustifier = lambda x: GMO(x, sigma=gmo_sigma)

    def forward(self, points, mesh_vertices, mesh_faces, return_distances=False):
        '''
        Given a point and a mesh surface, compute the distance to the closest point in the mesh surface. 
        
        Args:
            points: (batch_size, num_points, 3)
            mesh_vertices: (batch_size, num_vertices, 3)
            mesh_faces: (num_faces, 3)
        '''

        batch_size, num_points, _ = points.shape
        mesh_face_vertices = index_vertices_by_faces(mesh_vertices, mesh_faces) # (batch_size, num_faces, num_vertices, 3)

        # Get closest mesh triangles for each point
        _, face_idx, _ = point_to_mesh_distance(points, mesh_face_vertices)
        closest_triangle_vertices = torch.gather(mesh_face_vertices, dim=1, index=face_idx[:, :, None, None].expand(-1, -1, 3, 3)).view(batch_size, num_points, 3, 3)

        # Compute barycentric embedding of every point into the closest triangle
        a, b, c = closest_triangle_vertices[:, :, 0, :], closest_triangle_vertices[:, :, 1, :], closest_triangle_vertices[:, :, 2, :]   # (batch_size, num_points, 3)
        bcoords = PointToSurfaceLoss.barycentric_coords(a, b, c, points)    # (batch_size, num_points, 3)

        closest_points = torch.multiply(a, bcoords[:, :, 0].unsqueeze(-1)) + \
                        torch.multiply(b, bcoords[:, :, 1].unsqueeze(-1)) + \
                        torch.multiply(c, bcoords[:, :, 2].unsqueeze(-1))   # (batch_size, num_points, 3)
        square_distances = (closest_points - points).square().sum(-1)
        robust_distances = self.robustifier(square_distances)
        distance_loss = robust_distances.mean(-1).mean()

        if return_distances:
            return distance_loss, robust_distances
        else:
            return distance_loss

    @staticmethod
    def barycentric_coords(a, b, c, q, eps=1e-16):
        '''
        Compute Barycentric coordinates of q projected on the triangle spanned by the vertices a, b, and c
        '''

        v1 = b-a
        v2 = c-a
        n = torch.cross(v1, v2)
        n_dot_n = torch.sum(n*n, dim=-1)
        n_dot_n[n_dot_n < eps] = 1.0
        
        w = q - a
        gamma = torch.sum(torch.cross(v1, w)*n, dim=-1)/n_dot_n
        beta = torch.sum(torch.cross(w, v2)*n, dim=-1)/n_dot_n
        alpha = 1.0-beta-gamma
        return torch.stack((alpha, beta, gamma), dim=-1)

