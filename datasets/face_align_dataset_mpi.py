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

import os
import glob
import random
import imageio
from skimage.transform import rescale, resize

import numpy as np
import torch
import torch.utils.data as data

from psbody.mesh import Mesh
from utils import mesh_sampling, utils
from utils.camera import load_mpi_camera, rotate_image
from utils.data_augment import get_random_crop_offsets, scale_crop
from utils.utils import get_filename


class FaceAlignDatasetMPI(data.Dataset):
    def __init__(self, 
                data_list_fname,
                dataset_root_dir='',
                image_dir='',
                image_resize_factor = 1,
                calibration_dir='',
                scan_dir='',
                registration_root_dir='',   
                global_registration_root_dir='',
                mesh_sampler=None,
                # data augmentation parameters
                scale_min=0.9, # random scaling
                scale_max=1.1,
                brightness_sigma=0.1 / 3.0, # random brightness perturbation  
                scan_vertex_count=10000,
                # parameters to specify the type of images being loaded
                load_stereo_images=True,
                load_color_images=False,
                calibration_blacklist=[],
                image_file_ext='png'
                ):
        super().__init__()
       
        if os.path.exists(data_list_fname):
            self.split_list = utils.load_json(data_list_fname)
        else:
            raise RuntimeError('Invalid data path - %s' % data_list_fname)

        self.load_stereo_images = load_stereo_images
        self.load_color_images = load_color_images
        self.calibration_blacklist = calibration_blacklist

        # augmentation
        self.scale_min = scale_min # random scaling
        self.scale_max = scale_max
        self.brightness_sigma = brightness_sigma # random brightness perturbation

        self.mesh_sampler = mesh_sampler
        self.scan_vertex_count = scan_vertex_count
        self.registration_root_dir = registration_root_dir

        if os.path.exists(image_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(image_dir, subject, sequence, frame)
            self.img_fname = lambda subject, sequence, frame, view : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
        elif os.path.exists(dataset_root_dir):
            self.img_dir = lambda subject, sequence, frame : os.path.join(dataset_root_dir, subject, sequence, 'images', frame)
            self.img_fname = lambda subject, sequence, frame, view : os.path.join(self.img_dir(subject, sequence, frame), '%s.%s.%s.%s' % (sequence, frame, view, image_file_ext))
        else:
            raise RuntimeError('Invalid image directory')

        if os.path.exists(calibration_dir):
            self.calibration_dir = lambda subject, sequence : os.path.join(calibration_dir, subject, sequence)
        elif os.path.exists(dataset_root_dir):
            self.calibration_dir = lambda subject, sequence : os.path.join(dataset_root_dir, subject, sequence,  'meshes', '*')
        else:
            raise RuntimeError('Invalid calibration directory')

        self.scan_fname = ''
        if os.path.exists(scan_dir):
            self.scan_fname = lambda subject, sequence, frame : os.path.join(scan_dir, subject, sequence, '%s.%s.npy' % (sequence, frame))
        elif os.path.exists(dataset_root_dir):
            self.scan_fname = lambda subject, sequence, frame : os.path.join(dataset_root_dir, subject, sequence, 'meshes', '%s.%s.obj' % (sequence, frame))
             
        self.registration_fname = ''
        if os.path.exists(registration_root_dir):
            self.registration_fname = lambda subject, sequence, frame : os.path.join(registration_root_dir, subject, sequence, '%s.%s.ply' % (sequence, frame))

        self.global_mesh_fname = ''
        if os.path.exists(global_registration_root_dir):
            self.global_mesh_fname = lambda subject, sequence, frame : os.path.join(global_registration_root_dir, subject, sequence, '%s.%s.ply' % (sequence, frame))

        self.image_resize_factor = image_resize_factor
        self.data_size = len(self.split_list)

        # normalization
        # standard values from resnet:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
        self.mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = torch.from_numpy(self.mean_np)
        self.std  = torch.from_numpy(self.std_np)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.read(index % self.data_size)

    def read(self, index):
        to_meters = False
        subject, sequence, frame = self.split_list[index]

        # Read calibration files
        calib_fnames = sorted(glob.glob(os.path.join(self.calibration_dir(subject, sequence), '*.tka')))
        # print('Num calib fnames: %d' % len(calib_fnames))

        # Read stereo and color images for each calibration file
        color_images = []
        color_camera_intrinsics = []
        color_camera_extrinsics = []
        color_camera_distortions = []
        color_camera_centers = []
        color_images_augmented = []
        color_camera_intrinsics_augmented = []

        stereo_images = []
        stereo_camera_intrinsics = []
        stereo_camera_extrinsics = []
        stereo_camera_distortions = []
        stereo_camera_centers = []
        stereo_images_augmented = []
        stereo_camera_intrinsics_augmented = []

        for calib_fname in calib_fnames:
            # print('Process calib fname %s' % calib_fname)
            if (not self.load_stereo_images) and ('_A' in get_filename(calib_fname) or '_B' in get_filename(calib_fname)):
                continue
            if (not self.load_color_images) and ('_C' in get_filename(calib_fname)):
                continue
            if get_filename(calib_fname) in self.calibration_blacklist:
                continue

            img_with_camera = self.read_img_with_camera(subject, sequence, frame, calib_fname, to_meters=to_meters)

            # print(img_with_camera.keys())
            if img_with_camera is not None:
                if '_C' in get_filename(calib_fname):
                    color_images.append(img_with_camera['image'])
                    color_camera_intrinsics.append(img_with_camera['intrinsics'])
                    color_camera_extrinsics.append(img_with_camera['extrinsics'])
                    color_camera_distortions.append(img_with_camera['radial_distortion'])
                    color_camera_centers.append(img_with_camera['camera_center'])
                    color_images_augmented.append(img_with_camera['image_augmented'])
                    color_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])
                else:
                    stereo_images.append(img_with_camera['image'])
                    stereo_camera_intrinsics.append(img_with_camera['intrinsics'])
                    stereo_camera_extrinsics.append(img_with_camera['extrinsics'])
                    stereo_camera_distortions.append(img_with_camera['radial_distortion'])   
                    stereo_camera_centers.append(img_with_camera['camera_center'])         
                    stereo_images_augmented.append(img_with_camera['image_augmented'])
                    stereo_camera_intrinsics_augmented.append(img_with_camera['intrinsics_augmented'])                    
                
        if len(stereo_images) > 0:
            stereo_images = torch.stack(stereo_images, dim=0)
            stereo_camera_intrinsics = torch.stack(stereo_camera_intrinsics, dim=0)
            stereo_camera_extrinsics = torch.stack(stereo_camera_extrinsics, dim=0)
            stereo_camera_distortions = torch.stack(stereo_camera_distortions, dim=0)
            stereo_camera_centers = torch.stack(stereo_camera_centers, dim=0)
            stereo_images_augmented = torch.stack(stereo_images_augmented, dim=0)
            stereo_camera_intrinsics_augmented = torch.stack(stereo_camera_intrinsics_augmented, dim=0)

        if len(color_images) > 0:
            color_images = torch.stack(color_images, dim=0)
            color_camera_intrinsics = torch.stack(color_camera_intrinsics, dim=0)
            color_camera_extrinsics = torch.stack(color_camera_extrinsics, dim=0)
            color_camera_distortions = torch.stack(color_camera_distortions, dim=0)
            color_camera_centers = torch.stack(color_camera_centers, dim=0)
            color_images_augmented = torch.stack(color_images_augmented, dim=0)
            color_camera_intrinsics_augmented = torch.stack(color_camera_intrinsics_augmented, dim=0)

        data = {
            # img
            'color_images': color_images,
            'stereo_images': stereo_images,
            'color_images_augmented': color_images_augmented,
            'stereo_images_augmented': stereo_images_augmented,

            # camera
            'color_camera_intrinsics': color_camera_intrinsics,
            'color_camera_extrinsics': color_camera_extrinsics,
            'color_camera_distortions': color_camera_distortions,
            'color_camera_centers': color_camera_centers,
            'stereo_camera_intrinsics': stereo_camera_intrinsics,
            'stereo_camera_extrinsics': stereo_camera_extrinsics,
            'stereo_camera_distortions': stereo_camera_distortions,
            'stereo_camera_centers': stereo_camera_centers,
            
            'color_camera_intrinsics_augmented': color_camera_intrinsics_augmented,
            'stereo_camera_intrinsics_augmented': stereo_camera_intrinsics_augmented,

            # meta
            'index': index,
            'subject': subject,
            'sequence': sequence,
            'frame': frame,   
        }

        if (self.scan_fname != '') and (self.scan_vertex_count > 0):
            scan_fname = self.scan_fname(subject, sequence, frame)
            v_sampled = self.load_scan_vertices(scan_fname)
            data['v_scan'] = torch.from_numpy(np.array(v_sampled).astype(np.float32))

        # Load registration
        if self.registration_fname != '':
            registration_fname = self.registration_fname(subject, sequence, frame)
        
            if os.path.exists(registration_fname):
                data['registration_fname'] = registration_fname
                try:
                    registration = Mesh(filename=registration_fname)
                    if not to_meters:
                        registration.v[:] *= 1000 # FLAME registrations are in meters, if to_meters is false, convert them to milimeters
                except:
                    print(f'Unable to load registration {registration_fname}')

                v_registration, f_registration = registration.v, registration.f
                data['v_registration'] = torch.from_numpy(v_registration.astype(np.float32))
                data['f_registration'] = torch.from_numpy(f_registration.astype(np.int64))

                if self.mesh_sampler is not None:
                    for level in range(1,self.mesh_sampler.get_number_levels()):
                        v_registration, f_registration = self.mesh_sampler.downsample(v_registration, return_faces=True)

                data['v_reg_sampled'] = torch.from_numpy(v_registration.astype(np.float32))
                data['f_reg_sampled'] = torch.from_numpy(f_registration.astype(np.int64))

        if self.global_mesh_fname != '':
            global_mesh_fname = self.global_mesh_fname(subject, sequence, frame)
            if not os.path.exists(global_mesh_fname):
                print(f'Global mesh not found {global_mesh_fname}')

            try:
                global_mesh = Mesh(filename=global_mesh_fname)
                if not to_meters:
                    global_mesh.v[:] *= 1000 # FLAME registrations are in meters, if to_meters is false, convert them to milimeters
            except:
                print(f'Unable to load global mesh {global_mesh_fname}')
            data['v_reg_global'] = torch.from_numpy(global_mesh.v.astype(np.float32))
            data['f_reg_global'] = torch.from_numpy(global_mesh.f.astype(np.int64)) 
        else:
            # If no data from the global stage are provided, use the downsampled registrations as global stage initialization. 
            # Otherwise, load the global meshes.      
            if 'v_reg_sampled' in data:
                data['v_reg_global'] = data['v_reg_sampled']
                data['f_reg_global'] = data['f_reg_sampled']
            elif 'v_registration' in data:
                data['v_reg_global'] = data['v_registration']
                data['f_reg_global'] = data['f_registration']
        return data

    def load_scan_vertices(self, scan_fname):
        if not os.path.exists(scan_fname):
            raise RuntimeError(f'Scan not found {scan_fname}')

        file_extension = utils.get_extension(scan_fname)
        if file_extension.lower() in ['.obj', '.ply']:
            try:
                scan = Mesh(filename=scan_fname)
            except:
                raise RuntimeError(f'Unable to load scan {scan_fname}')

            import trimesh
            tr_mesh = trimesh.Trimesh(vertices=scan.v, faces=scan.f)
            v_sampled, _ = trimesh.sample.sample_surface(tr_mesh, self.scan_vertex_count)
            return v_sampled
        elif file_extension.lower() in ['.npy']:
            v_sampled = np.load(scan_fname)
            scan_v_ids = np.arange(v_sampled.shape[0])
            random.shuffle(scan_v_ids)
            scan_v_ids = scan_v_ids[:np.min((v_sampled.shape[0], self.scan_vertex_count))]
            return v_sampled[scan_v_ids]
        else:
            raise RuntimeError(f'Unknown scan file extension {file_extension}')

    def read_img_with_camera(self, subject, sequence, frame, calib_fname, to_meters=True):
        image_fname = self.img_fname(subject, sequence, frame, get_filename(calib_fname))
        if not os.path.exists(image_fname):
            print('Image file not found - %s' % image_fname)
            return None

        try:
            image = imageio.imread(image_fname, pilmode='RGB').astype(np.float32) / 255.
        except:
            raise RuntimeError('Error loading image - %s' % image_fname)

        camera = load_mpi_camera(calib_fname, self.image_resize_factor, to_meters=to_meters)
        if camera is None:
            return None

        # if self.image_resize_factor != 1:
        if (image.shape[0] != camera['image_size'][0]) or (image.shape[1] != camera['image_size'][1]):
            image = resize(image, (camera['image_size'][0], camera['image_size'][1]), anti_aliasing=True)

        if camera['image_size'][0] > camera['image_size'][1]:
            # The dataset contains images of landscape and portrait images of resolutions (A x B) and (B x A). 
            # To unify the images for batch handling, rotate all portrait images to landscape.
            image, camera = rotate_image(image, camera)
   
        # geometric augmentation by random scaling and cropping
        np.random.seed()
        crop_size = (camera['image_size'][0], camera['image_size'][1])
        scale_factor = self.scale_min + (self.scale_max - self.scale_min) * np.random.random()
        h_offset, w_offset = get_random_crop_offsets(crop_size, height=camera['image_size'][0], width=camera['image_size'][1])
        image_augmented, intrinsics_augmented = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'])

        # random brightness perturbation
        perturb = 1.0 + self.brightness_sigma * np.random.randn(1,1,3)
        image_augmented = image_augmented * perturb
        image_augmented = np.clip(image_augmented, 0., 1.)

        # normalize rgb
        image = self.normalize_image(image)
        image_augmented = self.normalize_image(image_augmented)

        image = torch.FloatTensor(torch.from_numpy(image.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
        intrinsics = torch.FloatTensor(torch.from_numpy(camera['intrinsics'].astype(np.float32)))
        extrinsics = torch.FloatTensor(torch.from_numpy(camera['extrinsics'].astype(np.float32)))
        radial_distortion = torch.FloatTensor(torch.from_numpy(camera['radial_distortion'].astype(np.float32)))
        camera_center = torch.FloatTensor(torch.from_numpy(camera['camera_center'].astype(np.float32)))

        image_augmented = torch.FloatTensor(torch.from_numpy(image_augmented.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
        intrinsics_augmented = torch.FloatTensor(torch.from_numpy(intrinsics_augmented.astype(np.float32)))

        return {
                    'image': image, 
                    'image_fname': image_fname, 
                    'intrinsics': intrinsics, 
                    'extrinsics': extrinsics, 
                    'radial_distortion': radial_distortion,
                    'camera_center': camera_center,
                    #augmented images
                    'image_augmented': image_augmented,
                    'intrinsics_augmented': intrinsics_augmented
                }

    # -----------------------
    # normalize input

    def normalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean_np.reshape((1,1,3)) ) / self.std_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return ( image - self.mean.view(1,3,1,1).to(image.device) ) / self.std.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")

    def denormalize_image(self, image):
        # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
        if isinstance(image, np.ndarray):
            if image.ndim !=3 or image.shape[2] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std_np.reshape((1,1,3)) + self.mean_np.reshape((1,1,3))
        elif torch.is_tensor(image):
            if image.ndimension() !=4 or image.shape[1] != 3:
                raise RuntimeError(f'invalid image shape {image.shape}')
            else:
                return image * self.std.view(1,3,1,1).to(image.device) + self.mean.view(1,3,1,1).to(image.device)
        else:
            raise RuntimeError(f"unrecognizable image type {type(image)}")
