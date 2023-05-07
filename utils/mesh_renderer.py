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

from psbody.mesh import Mesh

import os
if not os.environ.get( 'PYOPENGL_PLATFORM' ):
    # os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['PYOPENGL_PLATFORM'] = 'egl' 

import cv2
import imageio
import trimesh
import pyrender
import numpy as np

def dist_to_rgb(errors, min_dist=0.0, max_dist=1.0):
    import matplotlib as mpl
    import matplotlib.cm as cm
    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    cmap = cm.get_cmap(name='jet')
    colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return colormapper.to_rgba(errors)[:, 0:3]

def render_mesh(vertices, faces, vertex_colors=None, camera_extrinsics=None, camera_intrinsics=None, radial_distortion=None, 
                frustum={'near': 0.01, 'far': 3000.0}, image_size=(800, 800)):
    '''
    Returns a rendered mesh as uint8 tensor of size (H,W,3)
    '''

    vertices = np.copy(vertices)
    if camera_extrinsics is not None:
        num_points, _ = vertices.shape
        ones = np.ones((num_points, 1))        
        points_homogeneous = np.concatenate((vertices, ones), axis=-1)
        vertices = camera_extrinsics.dot(points_homogeneous.T).T

    z_coords = vertices[:,2].copy()
    z_coords[np.where(np.abs(z_coords) < 1e-7)] = 1.0
    vertices[:,0] = vertices[:,0] / z_coords
    vertices[:,1] = vertices[:,1] / z_coords
    vertices[:,2] = 1.0

    if radial_distortion is not None:
        K1, K2 = radial_distortion[0], radial_distortion[1]
        r2 = vertices[:,0]**2 + vertices[:,1]**2
        r4 = r2**2
        radial_distortion_factor = (1 + K1*r2 + K2*r4)
        vertices[:,0] = vertices[:,0]*radial_distortion_factor
        vertices[:,1] = vertices[:,1]*radial_distortion_factor    

    if camera_intrinsics is None:
        camera_intrinsics = np.array([
                [ 3000,      0,     image_size[0]//2 ],
                [    0,    3000,    image_size[1]//2 ],
                [    0,      0,        1.0 ]])

    vertices = camera_intrinsics.dot(vertices.T).T

    # Normalize the x and y coordinates
    vertices[:,0] = 2*(vertices[:,0] - (image_size[1]/2)) / image_size[0] 
    vertices[:,1] = 2*(vertices[:,1] - (image_size[0]/2)) / image_size[0]

    # Normalize z to be in [z_near, 2.0]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_diff = z_max-z_min
    if z_diff < 1e-7:
        z_diff = 1.0
    z_scale = (2.0-frustum['near'])/z_diff
    vertices[:,2] = z_scale*(z_coords-z_min)+frustum['near'] 

    rgb_per_v = 0.7*np.ones_like(vertices)
    if vertex_colors is not None:
        if vertex_colors.shape[0] == vertices.shape[0]:
            rgb_per_v[:] = vertex_colors

    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=frustum['near'], zfar=frustum['far'])
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    scene.add(camera, pose=camera_pose)

    angle = np.pi / 8.0
    pos = np.array([0,0,-2])
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=0.8)
    
    light_pose = np.eye(4)
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_height=image_size[0], viewport_width=image_size[1])
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame')
        print(e)
        color = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')
    return color
 
def render_textured_mesh(scan_fname, texture_fname, camera_extrinsics=None, camera_intrinsics=None, radial_distortion=None, 
                frustum={'near': 0.01, 'far': 3000.0}, image_size=(800, 800)):
    '''
    Returns a rendered mesh as uint8 tensor of size (H,W,3)
    '''

    tri_mesh = trimesh.load(scan_fname, process=False)
    # tri_mesh.vertices = 0.001*np.array(tri_mesh.vertices)
    texture = imageio.imread(texture_fname)

    vertices = np.copy(tri_mesh.vertices)
    if camera_extrinsics is not None:
        num_points, _ = vertices.shape
        ones = np.ones((num_points, 1))        
        points_homogeneous = np.concatenate((vertices, ones), axis=-1)
        vertices = camera_extrinsics.dot(points_homogeneous.T).T

    z_coords = vertices[:,2].copy()
    z_coords[np.where(np.abs(z_coords) < 1e-7)] = 1.0
    vertices[:,0] = vertices[:,0] / z_coords
    vertices[:,1] = vertices[:,1] / z_coords
    vertices[:,2] = 1.0

    if radial_distortion is not None:
        K1, K2 = radial_distortion[0], radial_distortion[1]
        r2 = vertices[:,0]**2 + vertices[:,1]**2
        r4 = r2**2
        radial_distortion_factor = (1 + K1*r2 + K2*r4)
        vertices[:,0] = vertices[:,0]*radial_distortion_factor
        vertices[:,1] = vertices[:,1]*radial_distortion_factor    

    if camera_intrinsics is None:
        camera_intrinsics = np.array([
                [ 3000,      0,     image_size[0]//2 ],
                [    0,    3000,    image_size[1]//2 ],
                [    0,      0,        1.0 ]])

    vertices = camera_intrinsics.dot(vertices.T).T

    # Normalize the x and y coordinates
    vertices[:,0] = 2*(vertices[:,0] - (image_size[1]/2)) / image_size[0] 
    vertices[:,1] = 2*(vertices[:,1] - (image_size[0]/2)) / image_size[0]

    # Normalize z to be in [z_near, 2.0]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_diff = z_max-z_min
    if z_diff < 1e-7:
        z_diff = 1.0
    z_scale = (2.0-frustum['near'])/z_diff
    vertices[:,2] = z_scale*(z_coords-z_min)+frustum['near'] 

    # material = trimesh.visual.texture.SimpleMaterial(image=texture)
    # visuals = trimesh.visual.TextureVisuals(uv=uv, image=texture, material=material)
    # tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visuals=visuals)
    # tri_mesh = trimesh.Trimesh(vertices=tri_mesh.vertices, faces=tri_mesh.faces, visuals=tri_mesh._visual, process=False)
    tri_mesh.vertices = vertices
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
    scene.add(render_mesh, pose=np.eye(4))

    camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=frustum['near'], zfar=frustum['far'])
    camera_pose = np.eye(4)
    camera_pose[:3,:3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    scene.add(camera, pose=camera_pose)

    angle = np.pi / 8.0
    pos = np.array([0,0,-2])
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=2.0)
    
    light_pose = np.eye(4)
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos) + np.array([0,0,1])
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_height=image_size[0], viewport_width=image_size[1])
        color, _ = r.render(scene, flags=flags)
    except Exception as e:
        print('pyrender: Failed rendering frame')
        print(e)
        color = np.zeros((image_size[0], image_size[1], 3), dtype='uint8')
    return color
 

# -----------------------------------------------------------------------------

def test_mesh_rendering():
    import glob
    import imageio
    from camera import load_mpi_camera

    resize_factor = 4

    mesh_fname = '/ps/data/Faces/FaceExploration/FaceExploration_221109_00201_TA/seq01_fear/meshes/seq01_fear.000001.obj'
    texture_fname = '/ps/data/Faces/FaceExploration/FaceExploration_221109_00201_TA/seq01_fear/meshes/seq01_fear.000001.bmp'

    mesh = trimesh.load(mesh_fname)
    texture = imageio.imread(texture_fname)

    calib_fnames = ['/ps/data/Faces/FaceExploration/FaceExploration_221109_00201_TA/seq01_fear/meshes/20221109100519237/31_C.tka']
    out_path = './results/render_test'
    os.makedirs(out_path, exist_ok=True)

    for calib_fname in calib_fnames:
        camera = load_mpi_camera(calib_fname, resize_factor=resize_factor)

        vertices = 1e-3*np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        uv = np.array(mesh.visual.uv)
        image_size = camera['image_size']

        camera_intrinsics = camera['intrinsics']
        camera_extrinsics = camera['extrinsics']
        radial_distortion = camera['radial_distortion']

        img = render_textured_mesh(vertices, faces, uv, texture, camera_extrinsics=camera_extrinsics, camera_intrinsics=camera_intrinsics, 
                            radial_distortion=radial_distortion, image_size=image_size)    
        imageio.imsave(os.path.join(out_path, '%s.jpg' % camera['name']), img)



    # mesh1_fname = '/ps/project/facealignments/FaMoS/neutral_align/FaMoS_180400_03331_TA/anger/anger.000001.ply'
    # mesh2_fname = '/ps/project/facealignments/FaMoS/neutral_align/FaMoS_180400_03331_TA/anger/anger.000052.ply'
    # mesh1 = Mesh(filename=mesh1_fname)
    # mesh2 = Mesh(filename=mesh2_fname)
    # vertex_dists = np.linalg.norm(mesh1.v-mesh2.v, axis=-1)
    # vertex_colors = dist_to_rgb(vertex_dists, min_dist=0.0, max_dist=0.01)

    # calib_fnames = sorted(glob.glob('/ps/data/Faces/FaMoS/FaMoS_180400_03331_TA/anger/meshes/20180420104318606/*.tka'))
    # out_path = './results/render_test'

    # for calib_fname in calib_fnames:
    #     camera = load_mpi_camera(calib_fname, resize_factor=resize_factor)

    #     vertices = mesh1.v
    #     faces = mesh1.f
    #     image_size = camera['image_size']

    #     camera_intrinsics = camera['intrinsics']
    #     camera_extrinsics = camera['extrinsics']
    #     radial_distortion = camera['radial_distortion']

    #     img = render_mesh(vertices, faces, vertex_colors=None, camera_extrinsics=camera_extrinsics, camera_intrinsics=camera_intrinsics, 
    #                         radial_distortion=radial_distortion, image_size=image_size)
    #     img2 = render_mesh(vertices, faces, vertex_colors=vertex_colors, camera_extrinsics=camera_extrinsics, camera_intrinsics=camera_intrinsics, 
    #                         radial_distortion=radial_distortion, image_size=image_size)
    
    #     imageio.imsave(os.path.join(out_path, '%s.jpg' % camera['name']), img)
    #     imageio.imsave(os.path.join(out_path, 'col_%s.jpg' % camera['name']), img2)

if __name__ == '__main__':
    test_mesh_rendering()
    print('Done')