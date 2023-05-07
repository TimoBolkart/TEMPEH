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
import kaolin
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MeshHelper:
    def __init__(self, num_vertices, faces):
        super().__init__()

        self.num_vertices = num_vertices
        self.faces = torch.from_numpy(faces.astype('int32')).long()
        self.vertices_by_faces = MeshHelper.vertex_by_face(num_vertices, faces)
        
    def vertex_normals(self, vertices):
        '''
        Compute the normalized vertex normals by averaging the normals of all faces adjacent to the vertex.
        Args:
            vertices: (B, L, 3), batched vertices
        Returns:
            vertex_normals: (B, L, 3), batched normalized vertex normals
        '''

        batch_size, num_vertices = vertices.shape[:2]
        device = vertices.device

        if num_vertices != self.num_vertices:
            raise RuntimeError(f"Wrong number of vertices: {num_vertices} != {self.num_vertices}")

        # Compute edges
        faces = self.faces.to(device)
        edges1 = torch.index_select(vertices, dim=1, index=faces[:, 1]) - \
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
        edges2 = torch.index_select(vertices, dim=1, index=faces[:, 2]) -\
                    torch.index_select(vertices, dim=1, index=faces[:, 0])
    
        # Compute face normals
        face_normals = torch.cross(edges1, edges2, dim=-1)

        # Compute vertex normals by averaging the normals of all adjacent faces
        vertices_by_faces = self.vertices_by_faces.to(device)
        vertex_normals = []
        for batch_idx in range(batch_size):
            vertex_normals.append(torch.matmul(vertices_by_faces, face_normals[batch_idx]))
        vertex_normals = torch.stack(vertex_normals).contiguous()
        return torch.nn.functional.normalize(vertex_normals, dim=-1)

    def vertex_visibility(self, vertices, camera_centers):
        '''
        Args:
            vertices: (B, L, 3), batched vertices
            camera_centers: (B, V, 3), batched camera centers for all V views
        Returns:
            visibilities: (B, V, L) batched binary vertex visibilities, with 0 = invisible, 1 = visible
        '''

        from psbody.mesh.visibility import visibility_compute

        batch_size, num_views, _ = camera_centers.shape
        num_vertices = vertices.shape[1]
        device = vertices.device

        if num_vertices != self.num_vertices:
            raise RuntimeError(f"Wrong number of vertices: {num_vertices} != {self.num_vertices}")

        np_vertices = vertices.detach().cpu().numpy().astype('float64')
        faces = self.faces.detach().cpu().numpy().astype('uint32')

        visibilities = []
        for batch_idx in range(batch_size):
            for view_idx in range(num_views):
                camera_center = camera_centers[batch_idx, view_idx].detach().cpu().numpy().astype('float64')              
                tmp_visibility, _ = np.array(visibility_compute(v=np_vertices[batch_idx], f=faces, cams=camera_center.reshape(1,3)))
                visibilities.append(tmp_visibility.squeeze())
        visibilities = np.stack(visibilities, axis=0).astype('float32').reshape((batch_size, num_views, num_vertices))
        return torch.from_numpy(visibilities).to(device)  

    def depth_vertex_visibility(self, vertices, camera_intrinsics, camera_extrinsics, depth_eps=3e-3, eps=1e-7, depth_rendering_size=200):
        '''
        Compute the vertex visibility with a thresholded depth rendering check. 

        Args:
            vertices: (B, L, 3), batched vertices
            camera_intrinsics: (B, V, 3, 3), batched camera intrinsics for all V views
            camera_extrinsics: (B, V, 3, 4), batched camera extrinsics for all V views
        Returns:
            visibilities: (B, V, L) batched binary vertex visibilities, with 0 = invisible, 1 = visible
        '''

        device = vertices.device
        batch_size, num_vertices, _ = vertices.shape
        num_faces, _ = self.faces.shape
        _, num_views, _, _ = camera_extrinsics.shape

        faces = self.faces.unsqueeze(0).repeat(batch_size*num_views,1,1).to(device) # (batch_size*num_views, num_faces, 3)
        camera_intrinsics = camera_intrinsics.contiguous().view(-1, 3, 3)   # (batch_size*num_views, 3, 3)
        camera_extrinsics = camera_extrinsics.contiguous().view(-1, 3, 4)   # (batch_size*num_views, 3, 4)

        vertices = vertices.transpose(1, 2) # (batch_size, 3, num_vertices)
        ones = torch.ones(batch_size, 1, num_vertices).to(device)   # (batch_size, 1, num_vertices)
        vertices_homogeneous = torch.cat((vertices, ones), axis=-2) # (batch_size, 4, num_vertices)

        vertices_homogeneous = vertices_homogeneous.unsqueeze(1).repeat(1,num_views,1,1).contiguous()    # (batch_size, num_views, 4, num_vertices)
        vertices_homogeneous = vertices_homogeneous.view(-1, 4, num_vertices)   # (batch_size*num_views, 4, num_vertices)

        # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
        vertices_image = camera_extrinsics.bmm(vertices_homogeneous)    # (batch_size*num_views, 3, num_vertices)
       
        vertices_depth = torch.clone(vertices_image[:,2,:])             # (batch_size*num_views, num_vertices)       
        min_depth, max_depth = torch.floor(vertices_depth.min(dim=-1).values), torch.ceil(vertices_depth.max(dim=-1).values)
        vertices_depth = (vertices_depth-min_depth.unsqueeze(-1))/(max_depth.unsqueeze(-1)-min_depth.unsqueeze(-1))

        # Transformation from 3D camera coordinate system to the undistorted image plane 
        mask = (vertices_image.abs() < eps)
        mask[:,:2,:] = False
        vertices_image[mask] = 1.0 # Avoid division by zero

        vertices_image[:,0,:] = vertices_image[:,0,:] / vertices_image[:,2,:]  
        vertices_image[:,1,:] = vertices_image[:,1,:] / vertices_image[:,2,:]  
        vertices_image[:,2,:] = 1.0

        # Transformation from image coordinates to the final image coordinates with the camera intrinsics
        vertices_image = camera_intrinsics.bmm(vertices_image)      # (batch_size*num_views, 3, num_vertices)    
        vertices_image = vertices_image.transpose(1,2)              # (batch_size*num_views, num_vertices, 3)
        vertices_image[:,:,2] = vertices_depth

        # Normalize image coordinates to [-1,1]
        x_min, x_max = torch.floor(vertices_image[:,:,0].min(dim=1).values), torch.ceil(vertices_image[:,:,0].max(dim=1).values)
        y_min, y_max = torch.floor(vertices_image[:,:,1].min(dim=1).values), torch.ceil(vertices_image[:,:,1].max(dim=1).values)
        x_center, y_center = (x_max+x_min)/2.0, (y_max+y_min)/2.0
        normalizing_scale = 1.0/torch.max(x_max-x_min, y_max-y_min)
        vertices_image[:,:,0] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,0]-x_center.unsqueeze(-1))
        vertices_image[:,:,1] = 2*normalizing_scale.unsqueeze(-1)*(vertices_image[:,:,1]-y_center.unsqueeze(-1))

        # Get vertices per face
        # faces = self.faces.unsqueeze(1).repeat(1,num_views,1,1).contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        faces = faces.contiguous().view(batch_size*num_views, num_faces*3) # (batch_size*num_views, num_faces*3)
        face_vertices_image = torch.gather(vertices_image, dim=1, index=faces[:,:].unsqueeze(-1).repeat(1,1,3)) # (batch_size*num_views, num_faces*3, 3)
        face_vertices_image = face_vertices_image.contiguous().view(-1, num_faces, 3, 3) # (batch_size*num_views, num_faces, 3, 3)

        # Render inverse depth images
        face_vertices_z = -face_vertices_image[:,:,:,-1]  # (batch_size, num_faces, 3)
        face_vertices_xy = torch.clone(face_vertices_image[:,:,:,:2]) # (batch_size, num_faces, 3, 2)
        face_vertices_xy[:,:,:,1] = -face_vertices_xy[:,:,:,1]
        face_features = 1.0-face_vertices_image[:,:,:,-1].unsqueeze(-1) # (batch_size, num_faces, 3, feature_dim)

        rendered_images, _ = kaolin.render.mesh.rasterize(depth_rendering_size, depth_rendering_size, face_vertices_z, face_vertices_xy, face_features)   
        rendered_images = rendered_images.contiguous().view(batch_size*num_views, depth_rendering_size, depth_rendering_size)  # (batch_size*num_views, image_size, image_size)

        # Sample depth images for every projected vertex
        img_grid = vertices_image[:,:,:2] # (batch_size*num_views, num_vertices, 2)
        img_grid = img_grid.unsqueeze(2) # (batch_size*num_views, num_vertices, 1, 2)
        sampled_image_inv_depths = F.grid_sample(rendered_images.view(-1, 1, depth_rendering_size, depth_rendering_size), img_grid, padding_mode='zeros', align_corners=False)    # (batch_size*num_views, 1, 1, num_vertices)
        sampled_image_inv_depths = sampled_image_inv_depths.contiguous().view(-1, num_vertices) # (batch_size*num_views, num_vertices)

        # Compute difference in depth between each sampled, projected vertex and the vertex.
        # If the sampled depth of the projected point is lower than the vertex depth, it is occluded.
        vertex_visibility_mask = vertices_depth-depth_eps <= 1.0-sampled_image_inv_depths
        vertex_visibility = vertex_visibility_mask.float()
        vertex_visibility = vertex_visibility.contiguous().view(batch_size, num_views, num_vertices)

        return vertex_visibility

    @staticmethod
    def vertex_by_face(num_vertices, faces):
        row = faces.flatten()
        col = np.array([range(faces.shape[0])] * 3).T.flatten()
        indices = np.stack((row, col)).tolist()
        data = [1.0] * col.shape[0]
        return torch.sparse_coo_tensor(indices, data, (num_vertices, faces.shape[0]),)

