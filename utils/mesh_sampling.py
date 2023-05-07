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

from audioop import reverse
import os
import math
import torch
import heapq
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from psbody.mesh import Mesh
from opendr.topology import get_vert_connectivity, get_vertices_per_edge, get_faces_per_edge

def get_nn(target_mesh, points):
    tree = KDTree(target_mesh.v)
    return np.concatenate(tree.query(points, k=1)[1], axis=0)

class MeshSampler(object):
    def __init__(self, mesh=None, mesh_dimension_list=[], keep_boundary_adjacent=True):
        super().__init__()
        self._number_levels = len(mesh_dimension_list)+1
        self._sampled_meshes = []
        self._adj_matrices = []
        self._trafos_down = []
        self._pt_trafos_down = []
        self._trafos_up = []
        self._pt_trafos_up = []
        self._vertices_per_level = []
        self._upsampled_vertex_ids = []
        if (mesh is not None) and (len(mesh_dimension_list)>0):
            self._generate_transform_matrices(mesh, mesh_dimension_list, keep_boundary_adjacent)
            self._generate_pt_matrices()
            self._compute_upsampled_vertex_ids()

    def save(self, fname):
        np.savez(fname, number_levels=self._number_levels, sampled_meshes=self._sampled_meshes, adj_matrices=self._adj_matrices,
                trafos_down=self._trafos_down, trafos_up=self._trafos_up, vertices_per_level=self._vertices_per_level,
                upsampled_vertex_ids=self._upsampled_vertex_ids)
    
    def load(self, fname):
        data = np.load(fname, allow_pickle=True)
        self._number_levels = int(data['number_levels'])
        self._sampled_meshes = list(data['sampled_meshes'])
        self._adj_matrices = list(data['adj_matrices'])
        self._trafos_down = list(data['trafos_down'])
        self._trafos_up = list(data['trafos_up'])
        self._vertices_per_level = list(data['vertices_per_level'])
        self._upsampled_vertex_ids = list(data['upsampled_vertex_ids'])
        self._generate_pt_matrices()

    def get_number_levels(self):
        return self._number_levels

    def get_level(self, num_vertices):
        return self._vertices_per_level.index(num_vertices)

    def get_mesh(self, level):
        if level >= -1 or level<self._number_levels:
            return self._sampled_meshes[level]
        else:
            return self._sampled_meshes[-1]

    def get_upsampled_vertex_ids(self, level):
        '''
        Get indices of the vertices added for a sampling level. 
        '''
        if level >=0 and level<self._number_levels-1:
            return self._upsampled_vertex_ids[level]
        return np.array([])
        
    def batch_downsample(self, vertices, return_faces=False):
        device = vertices.device
        batch_size, num_vertices, dim = vertices.shape
        # level = self._vertices_per_level.index(num_vertices)
        level = self.get_level(num_vertices)

        v_out = vertices
        f_out = self._sampled_meshes[level].f
        if level < self._number_levels-1:
            down_matrix = self._pt_trafos_down[level].to(device)
            vertices = vertices.transpose(0,1).reshape(num_vertices, -1)
            v_out = down_matrix.mm(vertices).reshape(-1, batch_size, dim).transpose(0,1)
            if return_faces:
                f_out = self._sampled_meshes[level+1].f
                
        if return_faces:
            return v_out, f_out
        else:
            return v_out

    def batch_upsample(self, vertices, return_faces=False):
        device = vertices.device
        batch_size, num_vertices, dim = vertices.shape
        # level = self._vertices_per_level.index(num_vertices)
        level = self.get_level(num_vertices)

        v_out = vertices
        f_out = self._sampled_meshes[level].f
        if level > 0:
            up_matrix = self._pt_trafos_up[level-1].to(device)
            vertices = vertices.transpose(0,1).reshape(num_vertices, -1)
            v_out = up_matrix.mm(vertices).reshape(-1, batch_size, dim).transpose(0,1)
            if return_faces:
                f_out = self._sampled_meshes[level-1].f

        if return_faces:
            return v_out, f_out
        else:
            return v_out

    def downsample(self, vertices, return_faces=False):
        num_vertices = vertices.shape[0]
        # level = self._vertices_per_level.index(num_vertices)
        level = self.get_level(num_vertices)

        v_out = vertices
        f_out = self._sampled_meshes[level].f
        if level < self._number_levels-1:
            v_out = self._trafos_down[level].dot(vertices)
            if return_faces:
                f_out = self._sampled_meshes[level+1].f
                
        if return_faces:
            return v_out, f_out
        else:
            return v_out

    def upsample(self, vertices, return_faces=False):
        num_vertices = vertices.shape[0]
        # level = self._vertices_per_level.index(num_vertices)
        level = self.get_level(num_vertices)
        v_out = vertices
        f_out = self._sampled_meshes[level].f
        if level > 0:
            v_out = self._trafos_up[level-1].dot(vertices)
            if return_faces:
                f_out = self._sampled_meshes[level-1].f
     
        if return_faces:
            return v_out, f_out
        else:
            return v_out

    def get_down_trafos(self, level=-1):
        if level < 0:
            return self._trafos_down
        elif level > 0 and level < len(self._trafos_down):
            return self._trafos_down[level]
        else:
            pass

    def get_up_trafos(self):
        return self._trafos_up

    def output_sampled_meshes(self, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        for mesh in self._sampled_meshes:
            mesh.write_obj(os.path.join(out_path, 'sampled_mesh_%04d.obj' % mesh.v.shape[0]))

    def _generate_transform_matrices(self, mesh, mesh_dimension_list, keep_boundary_adjacent):
        """Generates len(factors) meshes, each of them is scaled by factors[i] and
        computes the transformations between them.
        Parameters:
            mesh: template mesh
            factors: downsample factor (>=1, value 1 means no downsampling)

        Returns:
        M: a set of meshes downsampled from mesh by a factor specified in factors.
        A: Adjacency matrix for each of the meshes
        D: Downsampling transforms between each of the meshes
        U: Upsampling transforms between each of the meshes
        """

        # factors = map(lambda x: 1.0 / x, factors)
        self._adj_matrices.append(get_vert_connectivity(mesh.v, mesh.f))
        self._sampled_meshes.append(mesh)
        self._vertices_per_level.append(mesh.v.shape[0])

        # for factor in factors:
        mesh_dimension_list.sort(reverse=True)
        for num_vertices in mesh_dimension_list:    
            ds_f, ds_D, _ = self.qslim_decimator_transformer(self._sampled_meshes[-1], num_vertices=num_vertices, keep_boundary_adjacent=keep_boundary_adjacent)
            self._trafos_down.append(ds_D)

            new_mesh_v = ds_D.dot(self._sampled_meshes[-1].v)
            new_mesh = Mesh(v=new_mesh_v, f=ds_f)
            self._sampled_meshes.append(new_mesh)
            self._vertices_per_level.append(new_mesh.v.shape[0])
            self._adj_matrices.append(get_vert_connectivity(new_mesh.v, new_mesh.f))
            self._trafos_up.append(self.setup_deformation_transfer(self._sampled_meshes[-1], self._sampled_meshes[-2]))

    def _generate_pt_matrices(self):
        def csr_to_pt(csr_matrix):
            coo = csr_matrix.tocoo()
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            values = torch.FloatTensor(coo.data)
            return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape)) 

        for trafo_down in self._trafos_down:
            self._pt_trafos_down.append(csr_to_pt(trafo_down))

        for trafo_up in self._trafos_up:
            self._pt_trafos_up.append(csr_to_pt(trafo_up))            

    def _compute_upsampled_vertex_ids(self):
        '''
        Computer indices of the mesh vertices of a level that were added by upsampling from the previous level. 
        '''

        self._upsampled_vertex_ids = []
        for level in range(self._number_levels-1):
            current_mesh = self._sampled_meshes[level]
            next_mesh = self._sampled_meshes[level+1]
            nn_vertex_indices = get_nn(next_mesh, current_mesh.v)
            nn_distances = np.linalg.norm(current_mesh.v - next_mesh.v[nn_vertex_indices], axis=-1)
            upsampling_ids = np.where(nn_distances > np.finfo(np.float32).eps)[0]
            assert(upsampling_ids.shape[0] == current_mesh.v.shape[0]-next_mesh.v.shape[0])
            self._upsampled_vertex_ids.append(upsampling_ids)

    @staticmethod
    def qslim_decimator_transformer(mesh, factor=None, num_vertices=None, keep_boundary_adjacent=False):
        """Return a simplified version of this mesh.

        A Qslim-style approach is used here.

        :param factor: fraction of the original vertices to retain
        :param num_vertices: number of the original vertices to retain
        :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
        """

        if factor is None and num_vertices is None:
            raise Exception('Need either factor or num_vertices.')

        if num_vertices is None:
            num_vertices = math.ceil(len(mesh.v) * factor)

        Qv = MeshSampler.vertex_quadrics(mesh)

        # fill out a sparse matrix indicating vertex-vertex adjacency
        vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
        vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
                                shape=(len(mesh.v), len(mesh.v)))
        vert_adj = vert_adj + vert_adj.T
        vert_adj = vert_adj.tocoo()

        def collapse_cost(Qv, r, c, v):
            Qsum = Qv[r, :, :] + Qv[c, :, :]
            p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
            p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

            destroy_c_cost = p1.T.dot(Qsum).dot(p1)
            destroy_r_cost = p2.T.dot(Qsum).dot(p2)
            result = {
                'destroy_c_cost': destroy_c_cost,
                'destroy_r_cost': destroy_r_cost,
                'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
                'Qsum': Qsum}
            return result

        if keep_boundary_adjacent:
            v_boundary_ids = MeshSampler.get_mesh_boundary(mesh)
            v_boundary_adjacent_ids = MeshSampler.get_mesh_boundary_adjacent(mesh)

        # construct a queue of edges with costs
        queue = []
        for k in range(vert_adj.nnz):
            r = vert_adj.row[k]
            c = vert_adj.col[k]
            if r > c:
                continue

            cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
            if keep_boundary_adjacent:
                r_adjacent = r in v_boundary_adjacent_ids
                c_adjacent = c in v_boundary_adjacent_ids
                r_boundary = r in v_boundary_ids
                c_boundary = c in v_boundary_ids
                if r_adjacent and c_adjacent and (r_boundary != c_boundary):
                    cost = 1e10
            heapq.heappush(queue, (cost, (r, c)))

        # decimate
        collapse_list = []
        nverts_total = len(mesh.v)
        faces = mesh.f.copy()
        while nverts_total > num_vertices:
            e = heapq.heappop(queue)
            r = e[1][0]
            c = e[1][1]
            if r == c:
                continue

            cost = collapse_cost(Qv, r, c, mesh.v)
            if cost['collapse_cost'] > e[0]:
                heapq.heappush(queue, (cost['collapse_cost'], e[1]))
                # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
                continue
            else:
                # update old vert idxs to new one,
                # in queue and in face list
                if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                    to_destroy = c
                    to_keep = r
                else:
                    to_destroy = r
                    to_keep = c
                collapse_list.append([to_keep, to_destroy])

                # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
                np.place(faces, faces == to_destroy, to_keep)

                # same for queue
                which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
                which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
                for k in which1:
                    queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
                for k in which2:
                    queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

                Qv[r, :, :] = cost['Qsum']
                Qv[c, :, :] = cost['Qsum']

                a = faces[:, 0] == faces[:, 1]
                b = faces[:, 1] == faces[:, 2]
                c = faces[:, 2] == faces[:, 0]

                # remove degenerate faces
                def logical_or3(x, y, z):
                    return np.logical_or(x, np.logical_or(y, z))

                faces_to_keep = np.logical_not(logical_or3(a, b, c))
                faces = faces[faces_to_keep, :].copy()

            nverts_total = (len(np.unique(faces.flatten())))
        new_faces, mtx = MeshSampler.get_sparse_transform(faces, len(mesh.v))
        return new_faces, mtx, collapse_list

    @staticmethod
    def get_mesh_boundary(mesh):
        """Return indices of all boundary edge vertices (a boundary edge is defined as an edge only shared by one face
            :param mesh         input mesh
        """
        vpe = get_vertices_per_edge(mesh.v, mesh.f)
        fpe = get_faces_per_edge(mesh.v, mesh.f, verts_per_edge=vpe)
        boundary_edges = np.hstack((np.where(fpe[:, 0] == -1)[0], np.where(fpe[:, 1] == -1)[0])).ravel()
        return np.unique(vpe[boundary_edges, :].ravel())

    @staticmethod
    def get_mesh_boundary_adjacent(mesh):
        """Return indices of all vertices where one vertex is an edge vertex"""
        v_boundary = MeshSampler.get_mesh_boundary(mesh)
        vpe = get_vertices_per_edge(mesh.v, mesh.f)
        e_ids = np.where(np.logical_xor(np.isin(vpe[:,0], v_boundary), np.isin(vpe[:,1], v_boundary))==True)[0]
        return np.unique(vpe[e_ids].flatten())

    @staticmethod
    def vertex_quadrics(mesh):
        """Computes a quadric for each vertex in the Mesh.

        Returns:
        v_quadrics: an (N x 4 x 4) array, where N is # vertices.
        """

        # Allocate quadrics
        v_quadrics = np.zeros((len(mesh.v), 4, 4,))

        # For each face...
        for f_idx in range(len(mesh.f)):

            # Compute normalized plane equation for that face
            vert_idxs = mesh.f[f_idx]
            verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
            u, s, v = np.linalg.svd(verts)
            eq = v[-1, :].reshape(-1, 1)
            eq = eq / (np.linalg.norm(eq[0:3]))

            # Add the outer product of the plane equation to the
            # quadrics of the vertices for this face
            for k in range(3):
                v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)
        return v_quadrics

    @staticmethod
    def setup_deformation_transfer(source, target):
        nearest_faces, _, _ = source.compute_aabb_tree().nearest(target.v, True)
        nearest_faces = nearest_faces.ravel().astype(np.int64)

        v1 = source.v[source.f[nearest_faces][:,0]]
        v2 = source.v[source.f[nearest_faces][:,1]]
        v3 = source.v[source.f[nearest_faces][:,2]]
        b_coords = MeshSampler.barycentric_coords(v1, v2, v3, target.v)

        rows = np.tile(np.arange(target.v.shape[0]).reshape(-1,1), (1,3)).reshape(-1,)
        cols = source.f[nearest_faces].ravel()
        # coeffs_v = b_coords.ravel()
        coeffs = b_coords.ravel()
        # coeffs_n = np.zeros(3 * target.v.shape[0])

        # rows = np.hstack((rows, rows))  
        # cols = np.hstack((cols, source.v.shape[0]))
        # coeffs = np.hstack((coeffs_v, coeffs_n))  
        
        matrix = sp.csc_matrix((coeffs, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
        return matrix

    @staticmethod
    def barycentric_coords(a, b, c, q):
        '''Compute Barycentric coordinates of q projected on the triangle spanned by the vertices a, b, and c'''
        v1 = b-a
        v2 = c-a
        n = np.cross(v1, v2)
        n_dot_n = np.sum(n*n, axis=-1)
        w = q - a

        gamma = np.sum(np.cross(v1, w)*n, axis=-1)/n_dot_n
        beta = np.sum(np.cross(w, v2)*n, axis=-1)/n_dot_n
        alpha = 1.0-beta-gamma
        return np.vstack((alpha, beta, gamma)).T

    @staticmethod
    def get_sparse_transform(faces, num_original_verts):
        verts_left = np.unique(faces.flatten())
        IS = np.arange(len(verts_left))
        JS = verts_left
        data = np.ones(len(JS))

        mp = np.arange(0, np.max(faces.flatten()) + 1)
        mp[JS] = IS
        new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

        ij = np.vstack((IS.flatten(), JS.flatten()))
        mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))
        return (new_faces, mtx)
