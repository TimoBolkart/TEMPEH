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

from __future__ import print_function
import os
import sys
import numpy as np
from os.path import exists, join
import torch
import torch.nn as nn
import torch.nn.functional as F
from psbody.mesh import Mesh

def merge_meshes(mesh_list):
    v = mesh_list[0].v
    f = mesh_list[0].f
    i = v.shape[0]
    for m in mesh_list[1:]:
        v = np.vstack((v, m.v))
        f = np.vstack((f, i+m.f))
        i = v.shape[0]
    merged_mesh = Mesh(v, f)

    has_vc = hasattr(mesh_list[0], 'vc')
    if has_vc:
        vc_list = []
        for m in mesh_list:
            vc_list.append(m.vc)
        vc = np.vstack(vc_list)
        merged_mesh.set_vertex_colors(vc)
    return merged_mesh

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix

    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Copied from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_axis_angle

    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation by Zhou et al. [1] to axis/angle.
    
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """

    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))

def get_path(filename):
    return os.path.dirname(filename)

def get_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0] 

def get_extension(filename):
    return os.path.splitext(filename)[1] 

def get_sub_folder(path):
    if not os.path.isdir(path):
        return []
    sub_folders = []
    for sub_folder in os.listdir(path):
        sub_dir = os.path.join(path, sub_folder)
        if not os.path.isdir(sub_dir):
            continue
        else:
            sub_folders.append(sub_folder)
    return sub_folders

# -----------------------------------------------------------------------------

def print_memory(device, prefix=''):
    print('%s - Allocated memory: %f GB' % (prefix, torch.cuda.memory_allocated(device)/(1024*1024*1024)))

# -----------------------------------------------------------------------------

def to_numpy(input):
    if torch.is_tensor(input):
        return input.detach().cpu().numpy()
    else:
        return input

# -----------------------------------------------------------------------------

def get_time_string():
    from datetime import datetime
    mydate = datetime.now()
    return '%02d:%02d:%02d' % (mydate.hour, mydate.minute, mydate.second)

# -----------------------------------------------------------------------------

def load_binary_pickle( filepath ):
    if sys.version_info[0] < 3:
        import cPickle as pickle
    else:
        import pickle

    with open( filepath, 'rb' ) as f:
        data = pickle.load( f )
    return data

# -----------------------------------------------------------------------------

def save_binary_pickle( data, filepath ):
    if sys.version_info[0] < 3:
        import cPickle as pickle
    else:
        import pickle
    with open( filepath, 'wb' ) as f:
        pickle.dump( data, f )

# -----------------------------------------------------------------------------

def save_npy( data, filepath ):
    with open( filepath, 'wb' ) as fp:
        np.save( fp, data ) 

# -----------------------------------------------------------------------------

def load_npy( filepath ):
    data = None
    with open( filepath, 'rb' ) as fp:
        data = np.load( fp ) 
    return data

# -----------------------------------------------------------------------------

def load_json( filepath ):
    import json
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

# -----------------------------------------------------------------------------

def save_json(data, filepath, indent=4, verbose=False):
    import json
    with open(filepath, 'w') as fp:
        json.dump(data, fp, indent=indent)
    if verbose: print(f"saved json file at: {filepath}")

# -----------------------------------------------------------------------------

def get_extension( file_path ):
    import os.path
    return os.path.splitext( file_path )[1] # returns e.g. '.png'

# -----------------------------------------------------------------------------

def safe_mkdir( file_dir, enable_777=False, recursive=True ):
    if sys.version_info[0] < 3:
        if not os.path.exists( file_dir ):
            os.mkdir( file_dir )
    else:
        from pathlib import Path
        path = Path(file_dir)
        path.mkdir(parents=recursive, exist_ok=True)
    if enable_777:
        chmod_777( file_dir )

# -----------------------------------------------------------------------------

def value2color( data, vmin=0, vmax=0.001, cmap_name='jet' ):
    # 'data' is np.array in size (H,W)
    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.Normalize( vmin=vmin, vmax=vmax )
    cmap = cm.get_cmap( name=cmap_name )
    colormapper = cm.ScalarMappable( norm=norm, cmap=cmap )
    rgba = colormapper.to_rgba( data.astype(np.float) )
    color_3d = rgba[...,0:3]
    return color_3d

# -----------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        # assume val is the average value for a batch
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# -----------------------------------------------------------------------------

class AdvancedMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return np.sum(np.asarray(self.vals_weighted), axis=0) / np.asarray(self.bs).sum()

    @property
    def median(self):
        return np.median(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def max(self):
        return np.max(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def min(self):
        return np.min(np.asarray(self.vals_weighted), axis=0) # only works if bs = 1 for all entries

    @property
    def percentile25(self):
        return np.percentile(np.asarray(self.vals_weighted), 25, axis=0) # only works if bs = 1 for all entries

    @property
    def percentile75(self):
        return np.percentile(np.asarray(self.vals_weighted), 75, axis=0) # only works if bs = 1 for all entries

    @property
    def val(self):
        return self.vals[-1] # last value

    @property
    def sum(self):
        return np.sum(np.asarray(self.vals_weighted), axis=0)

    @property
    def count(self):
        return np.asarray(self.bs).sum()

    @property
    def records(self):
        return {
            'vals': self.vals.tolist() if isinstance(self.vals, np.ndarray) else self.vals,
            'vals_weighted': self.vals_weighted.tolist() if isinstance(self.vals_weighted, np.ndarray) else self.vals_weighted,
            'bs': self.bs.tolist() if isinstance(self.bs, np.ndarray) else self.bs
        }

    def reset(self):
        self.vals = []
        self.vals_weighted = []
        self.bs = []

    def update(self, val, n=1):
        # # assume val is the average value for a batch
        if isinstance(val, list):
            val_w = [ el * n for el in val ]
        elif isinstance(val, np.ndarray):
            val_w = ( val * n ).tolist()
            val = val.tolist()
        else:
            val_w = [ val * n ]
            val = [ val ]

        self.vals.append(val)
        self.vals_weighted.append(val_w)
        self.bs.append(n)