from bdb import set_trace
from fileinput import filename
import os
import torch
from pickletools import uint8
import numpy as np
from scipy.linalg import lstsq
from os.path import splitext, basename
from skimage.transform import rescale, resize, rotate

def convert_distortion_parms(k1, k2, fl, fx, fy, width, height):
    # OpenCV wants radial distortion parameters that are applied to image plane coordinates
    # prior to being scaled by fx and fy (so not pixel coordinates). In contrast, k1 and k2
    # are defined via Tsai camera calibration http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
    K1 = k1 * (fl ** 2.0)
    K2 = k2 * (fl ** 4.0)
    # Also, K1 and K2 are actually undistortion coefficients. They go from distorted to undistorted image
    # plane coordinates. OpenCV wants coefficients that go the other way.
    r_values = .01 * np.array(range(1, 101)) * (((width / fx) ** 2.0 + (height / fy) ** 2.0) ** 0.5)
    undistorted_r_values = r_values * (1 + K1 * (r_values ** 2.0) + K2 * (r_values ** 4.0))
    distortion_factors = r_values / undistorted_r_values
    # Given the undistorted and distorted distances, we solve for the new distortion factors via linear regression
    k1, k2 = lstsq(np.matrix([undistorted_r_values ** 2.0, undistorted_r_values ** 4.0]).T, np.matrix(distortion_factors - 1.0).T)[0]
    return (k1, k2)

def load_mpi_camera(calib_fname, resize_factor=1, to_meters=True):
    '''
    
    '''
    # load
    if not os.path.exists(calib_fname):
        print('Calibration file not found - %s' % calib_fname)
        return None

    with open(calib_fname, 'r') as fp:
        lines = fp.readlines()

    M = [
        [ float(el) for el in lines[1][:-2].split(' ') ],
        [ float(el) for el in lines[2][:-2].split(' ') ],
        [ float(el) for el in lines[3][:-2].split(' ') ],
    ]

    # Position
    X = float(lines[4][3:-1])
    Y = float(lines[5][3:-1])
    Z = float(lines[6][3:-1])
    # Focal length
    f = float(lines[7][3:-1])
    # Radial distortion
    K = float(lines[8][3:-1])
    # 2dn order radial distortion
    K2 = float(lines[9][4:-1])
    S = float(lines[10][3:-1])
    # Effective pixel size
    x = float(lines[11][3:-1])
    y = float(lines[12][3:-1])
    # Principal point
    a = float(lines[13][3:-1])
    b = float(lines[14][3:-1])
    # Image size (width, height)
    #image_size = [ int(el) for el in lines[15][4:-1].split(' ') ]
    image_width, image_height = [ int(el) for el in lines[15][4:-1].split(' ') ]
    c = float(lines[16][3:-1])

    camera_rot = np.array(M).reshape((3,3))
    camera_trans = np.array([X, Y, Z])
    focal_length = f
    principal_point = np.array([a, b])
    pixel_size = np.array([x, y])

    if resize_factor != 1:
        image_width, image_height = image_width // resize_factor, image_height // resize_factor
        focal_length /= resize_factor
        principal_point /= resize_factor

    # Transformation from undistorted image plane to distorted image coordinates
    K1, K2 = convert_distortion_parms(K, K2, focal_length, focal_length / pixel_size[0], focal_length / pixel_size[1], image_width, image_height)
    radial_distortion = np.array([K1, K2]).reshape(-1,)

    intrinsics = np.array([
        [focal_length/pixel_size[0],                 0,              principal_point[0] ],
        [              0,               focal_length/pixel_size[1],  principal_point[1] ],
        [              0,                            0,                      1.0        ]])

    camera_trans = -camera_rot.dot(camera_trans)
    if to_meters:
        camera_trans /= 1000.    
    extrinsics =  np.concatenate((camera_rot, camera_trans.reshape((3,1))), axis=1)

    cam2world_trafo = get_camera_to_world_transformation(extrinsics)
    center = cam2world_trafo[:3, 3]
    view_direction = cam2world_trafo[:3,:3].dot(np.array([0,0,1]))

    name = splitext(basename(calib_fname))[0]
    camera = {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'radial_distortion': radial_distortion, 
        'camera_center': center,
        'view_direction': view_direction,
        'image_size': np.array([image_height, image_width]),        
        'name': name
    }
    return camera

def get_camera_to_world_transformation(extrinsics):
    '''
    Get transformation matrix (3,4) that transforms points in the local camera coordinate systems
    to the world coordinate system.
    '''

    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return np.hstack((R.T, -R.T.dot(t)[:,np.newaxis]))

def rotate_image(image, camera=None):
    image = rotate(image, 90, resize=True)

    if camera is not None:
        Rt = np.array([ 
            [ 0,    1,             0           ],
            [-1,    0, camera['image_size'][1] ],
            [ 0,    0,             1           ]
        ])

        camera = dict(camera)
        camera['intrinsics'] = Rt.dot(camera['intrinsics'])
        camera['image_size'] = camera['image_size'][::-1]
        return image, camera
    else:
        return image
       
def crop_image(image, camera=None):
    pass

def scale_image(image, scale_factor, camera=None):
    img = rescale(image, scale_factor, channel_axis=2, anti_aliasing=True)
    if camera is None:
        return img
    else:
        scale_mat = np.eye(3)
        scale_mat[0,0] = scale_mat[1,1] = scale_factor
        camera['intrinsics'] = scale_mat.dot(camera['intrinsics'])
        return img, camera

def perspective_project(points, camera_intrinsics, camera_extrinsics, radial_distortion, eps=1e-7):
    '''
    Projection of 3D points into the image plane using a perspective transformation.
    :param points:      array of 3D points (num_points X 3)
    :return:            array of projected 2D points (num_points X 2)
    '''

    num_points, _ = points.shape
    ones = np.ones((num_points, 1))
    points_homogeneous = np.concatenate((points, ones), axis=-1)

    # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
    points_image = camera_extrinsics.dot(points_homogeneous.T).T

    # Transformation from 3D camera coordinate system to the undistorted image plane 
    z_coords = points_image[:,2]
    z_coords[np.where(np.abs(z_coords) < eps)] = 1.0
    points_image[:,0] = points_image[:,0] / z_coords
    points_image[:,1] = points_image[:,1] / z_coords

    # Transformation from undistorted image plane to distorted image coordinates
    K1, K2 = radial_distortion[0], radial_distortion[1]
    r2 = points_image[:,0]**2 + points_image[:,1]**2
    r4 = r2**2
    radial_distortion_factor = (1 + K1*r2 + K2*r4)
    points_image[:,0] = points_image[:,0]*radial_distortion_factor
    points_image[:,1] = points_image[:,1]*radial_distortion_factor    
    points_image[:,2] = 1.0

    # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
    points_image = camera_intrinsics.dot(points_image.T).T
    return points_image

def batch_perspective_project(points, camera_intrinsics, camera_extrinsics, radial_distortion, eps=1e-7):
    device = points.device
    batch_size, num_points, _ = points.shape
    points = points.transpose(1, 2)

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
    K1, K2 = radial_distortion[:,0], radial_distortion[:,1]       # (batch_size)
    r2 = points_image_x**2 + points_image_y**2            # (batch_size, num_points)
    r4 = r2**2
    radial_distortion_factor = (1 + K1[:, None]*r2 + K2[:, None]*r4)  # (batch_size, num_points)

    points_image_x = points_image_x*radial_distortion_factor
    points_image_y = points_image_y*radial_distortion_factor
    points_image_z = torch.ones_like(points_image[:,2,:])
    points_image = torch.cat((points_image_x[:, None, :], points_image_y[:, None, :], points_image_z[:, None, :]), dim=1)

    # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
    points_image = camera_intrinsics.bmm(points_image)              # (batch_size, 3, num_points)    
    points_image = torch.transpose(points_image, 1, 2)[:,:,:2]      # (batch_size, num_points, 2) 
    return points_image

