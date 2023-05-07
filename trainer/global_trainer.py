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
import time
import random
from os.path import join

import torch
import numpy as np
from torch.autograd import Variable
from utils.utils import print_memory, to_numpy, get_time_string
from utils.mesh_renderer import render_mesh, dist_to_rgb
from utils.data_augment import get_subset_views
from utils.point_to_point_loss import PointToPointLoss
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance
from utils.edge_loss import EdgeLoss

from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions

from psbody.mesh import Mesh
from utils.mesh_sampling import MeshSampler

# -----------------------------------------------------------------------------

class Trainer(BaseTrainer):
    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device
        self.vis_view_ids = None

    def register_mesh_sampler(self):
        mesh_sampler_fname = join(self.directory_output, 'mesh_sampler.npz')
        if os.path.exists(mesh_sampler_fname):
            self.mesh_sampler = MeshSampler()
            self.mesh_sampler.load(mesh_sampler_fname)
        else:
            template_fname = self.args.template_fname
            if not os.path.exists(template_fname):
                raise RuntimeError('Template not found %s' % template_fname)
            template_mesh = Mesh(filename=template_fname)
            template_mesh.v[:] *= 1000
            mesh_dimension_list = self.args.number_sample_points
            self.mesh_sampler = MeshSampler(template_mesh, mesh_dimension_list, keep_boundary_adjacent=True)
            self.mesh_sampler.save(mesh_sampler_fname)

    def register_model(self):
        import models.model_aligner.prototypes.model_global_stage as models
        model = models.Model(args=self.args)
        model.initialize(init_method='normal')
        if self.args.pretrained_path:
            raise RuntimeError('Not yet tested if loading is good')
        model = model.to(self.device)
        self.model = torch.nn.DataParallel(model)

    def register_dataset(self):
        train_data_list_fname = self.args.train_data_list_fname
        val_data_list_fname = self.args.val_data_list_fname
        dataset_root_dir = self.args.dataset_directory
        image_dir = self.args.image_directory
        calibration_dir = self.args.calibration_directory
        registration_root_dir = self.args.processed_directory
        scan_dir = self.args.scan_directory
        scan_vertex_count = self.args.scan_vertex_count

        image_resize_factor = self.args.image_resize_factor
        load_stereo_images = self.args.input_image_type == 'stereo_images'
        load_color_images = self.args.input_image_type == 'color_images'
        image_file_ext = self.args.image_file_ext

        from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI       
        self.dataset_train = FaceAlignDatasetMPI(   data_list_fname=train_data_list_fname,
                                                    dataset_root_dir=dataset_root_dir, 
                                                    image_dir=image_dir,
                                                    calibration_dir=calibration_dir,
                                                    scan_dir=scan_dir,
                                                    registration_root_dir=registration_root_dir,
                                                    image_resize_factor=image_resize_factor,
                                                    mesh_sampler=self.mesh_sampler,            
                                                    scan_vertex_count=scan_vertex_count,
                                                    brightness_sigma=self.args.brightness_sigma,
                                                    load_stereo_images=load_stereo_images,
                                                    load_color_images=load_color_images,
                                                    image_file_ext=image_file_ext)

        self.dataset_val = FaceAlignDatasetMPI( data_list_fname=val_data_list_fname,
                                                dataset_root_dir=dataset_root_dir, 
                                                image_dir=image_dir,    
                                                calibration_dir=calibration_dir,
                                                scan_dir=scan_dir,
                                                registration_root_dir=registration_root_dir,
                                                image_resize_factor=image_resize_factor,           
                                                mesh_sampler=self.mesh_sampler,
                                                scan_vertex_count=scan_vertex_count,
                                                brightness_sigma=self.args.brightness_sigma,
                                                load_stereo_images=load_stereo_images,
                                                load_color_images=load_color_images,
                                                image_file_ext=image_file_ext)

        self.dataloader_train = self.make_data_loader(self.dataset_train, cuda=True, shuffle=True)
        self.dataloader_val = self.make_data_loader(self.dataset_val, cuda=True, shuffle=True)

    def register_losses(self):
        vertex_masks = None
        if os.path.exists(self.args.vertex_mask_fname):
            vertex_masks = np.load(self.args.vertex_mask_fname)
        sample_mesh = self.mesh_sampler.get_mesh(-1)

        if self.args.point2point_loss_function in ['MSE', 'mse']:
            p2p_loss = torch.nn.MSELoss()
        elif self.args.point2point_loss_function in ['Wing', 'wing']:
            from utils.wing_loss import WingLoss
            p2p_loss = WingLoss()
        else:
            raise RuntimeError( "unrecognizable point2point_loss_function: %s" % ( self.point2point_loss_function ) )

        self.points_loss_function = PointToPointLoss(num_vertices=sample_mesh.v.shape[0],
                                                    vertex_masks=vertex_masks, mask_weights=self.args.point_mask_weights,
                                                    mesh_sampler=self.mesh_sampler,
                                                    loss_function=p2p_loss)
        self.points2surface_loss_function = PointToSurfaceLoss(gmo_sigma=self.args.gmo_sigma)
        self.edge_loss_function = EdgeLoss( num_vertices=sample_mesh.v.shape[0], faces=sample_mesh.f, 
                                            vertex_masks=vertex_masks, mask_weights=self.args.edge_mask_weights,
                                            mesh_sampler=self.mesh_sampler)

    def feed_data(self, data):
        if self.args.input_image_type == 'stereo_images':
            number_views = data['stereo_images'].shape[1]
            images = data['stereo_images_augmented']
            camera_intrinsics = data['stereo_camera_intrinsics_augmented']
            camera_extrinsics = data['stereo_camera_extrinsics']
            camera_distortions = data['stereo_camera_distortions']
        elif self.args.input_image_type == 'color_images':
            number_views = data['color_images'].shape[1]
            images = data['color_images_augmented']
            camera_intrinsics = data['color_camera_intrinsics_augmented']
            camera_extrinsics = data['color_camera_extrinsics']
            camera_distortions = data['color_camera_distortions']            
        else:
            raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

        if self.args.sample_views and self.model.training:
            views = get_subset_views(number_views, minimum_views=self.args.minimum_sample_views)
        else:
            views = np.arange(number_views)

        self.data = data
        self.inputs = {
            'images': Variable(images[:,views,...]).to(self.device),
            'camera_intrinsics': Variable(camera_intrinsics[:,views,...]).to(self.device),
            'camera_extrinsics': Variable(camera_extrinsics[:,views,...]).to(self.device),
            'camera_distortions': Variable(camera_distortions[:,views,...]).to(self.device)
        }

        self.target_vertices = data['v_registration'].to(self.device)

    def forward(self):
        random_grid = True if self.model.training else False
        self.global_points, _, self.global_grid, self.global_grid_origin, self.global_grid_scales = self.model(**self.inputs, random_grid=random_grid)

        if self.global_step % self.args.print_frequency == 0:        
            print_memory(self.device, prefix='FW')

    def compute_losses(self):
        self.points_loss, self.points2surface_loss, self.edge_regularizer_loss = 0.0, 0.0, 0.0

        if self.args.weight_points_recon > 0.0:
            self.points_loss = self.args.weight_points_recon*self._points_reconstruction_loss()
        if self.args.weight_points2surface > 0.0:
            self.points2surface_loss = self.args.weight_points2surface*self._points2surface_loss()
        if self.args.weight_edge_regularizer > 0.0:
            self.edge_regularizer_loss = self.args.weight_edge_regularizer*self._edge_regularizer()
        self.loss = self.points_loss + self.points2surface_loss + self.edge_regularizer_loss

    def backward(self):
        self.optimizer_model.zero_grad()
        self.loss.backward()
        if self.args.gradient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.module.optimizable_parms(), max_norm=self.args.gradient_max_norm, norm_type=2)
        self.optimizer_model.step()

    def run(self):
        batch_size = self.args.batch_size
        num_epoch = int(np.ceil(self.args.num_iterations / float(len(self.dataset_train)) * batch_size))
        start_epoch = int(self.global_step / float(len(self.dataset_train)) * batch_size)+1
        print("expect to run for %d epoches" % (num_epoch-start_epoch))

        for epoch_idx in range(start_epoch, num_epoch+1):
            np.random.seed() # reset seed
            print('************************')
            print('Epoch %d / %d' % (epoch_idx, num_epoch))
            print('************************')
            self.train_one_epoch()

    def train_one_epoch(self):
        for data in self.dataloader_train:
            self.train_step(data)

    def train_step(self, data):
        lr = self.adjust_learning_rate(self.global_step, self.args.num_iterations, method=self.args.lr_type)             
        self.set_train()
        self.feed_data(data)
        self.forward()
        self.compute_losses()
        self.backward()

        if self.global_step % self.args.print_frequency == 0:        
            print('%s, step %d, total loss: %f, learning rate: %f' %(get_time_string(), self.global_step, to_numpy(self.loss), lr))
            self.tb_logger.add_scalar('Learning rate/train', lr, self.global_step)
            self.tb_logger.add_scalar('Total loss/train', to_numpy(self.loss), self.global_step)
            self.tb_logger.add_scalar('Points loss/train', to_numpy(self.points_loss), self.global_step)
            self.tb_logger.add_scalar('Points2Surface loss/train', to_numpy(self.points2surface_loss), self.global_step)
            self.tb_logger.add_scalar('Edge regularizer loss/train', to_numpy(self.edge_regularizer_loss), self.global_step)     
            self.tb_logger.add_scalar('Points2Surface distance/train', to_numpy(self._points2surface_loss(no_robustifier=True)), self.global_step)        
        if (self.global_step > 0) and (self.global_step % self.args.visualize_frequency == 0):
            self.visualize('train')
        if (self.global_step > 0) and (self.global_step % self.args.validate_frequency == 0):               
            self.validate()
        if (self.global_step > 0) and (self.global_step % self.args.save_frequency == 0):
            self.save_checkpoint()
        self.global_step += 1

    def validate(self):
        self.set_eval()
        with torch.no_grad():
            total_val_loss = []
            total_points_loss = []
            total_points2surface_loss = []
            total_edge_reg_loss = []
            total_points2surface_distance = []
            with torch.no_grad():
                for data in self.dataloader_val:
                    self.feed_data(data)
                    self.forward()
                    self.compute_losses()
                    total_val_loss.append(to_numpy(self.loss))
                    total_points_loss.append(to_numpy(self.points_loss))
                    total_points2surface_loss.append(to_numpy(self.points2surface_loss))
                    total_edge_reg_loss.append(to_numpy(self.edge_regularizer_loss))
                    total_points2surface_distance.append(to_numpy(self._points2surface_loss(no_robustifier=True)))            
            self.tb_logger.add_scalar('Total loss/validation', np.mean(np.array(total_val_loss)), self.global_step)
            self.tb_logger.add_scalar('Points loss/validation', np.mean(np.array(total_points_loss)), self.global_step)
            self.tb_logger.add_scalar('Points2Surface loss/validation', np.mean(np.array(total_points2surface_loss)), self.global_step)
            self.tb_logger.add_scalar('Edge regularizer loss/validation', np.mean(np.array(total_edge_reg_loss)), self.global_step)                              
            self.tb_logger.add_scalar('Points2Surface distance/validation', np.mean(np.array(total_points2surface_distance)), self.global_step)        
            self.visualize('val')

    def visualize(self, mode='train'):
        if self.vis_view_ids is None:
            if self.args.input_image_type == 'stereo_images':
                num_views = len(self.data['stereo_images'][0])     
            elif self.args.input_image_type == 'color_images':
                num_views = len(self.data['color_images'][0])     
            else:
                raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

            view_ids = np.arange(num_views)
            random.Random(7).shuffle(view_ids)
            self.vis_view_ids = view_ids[:6]
        
        with torch.no_grad():
            target_vertices = to_numpy(self.data['v_registration'][0])
            reconstructed_vertices = to_numpy(self.global_points[0])
            faces = to_numpy(self.data['f_registration'][0])

            vertex_distance = np.linalg.norm(target_vertices-reconstructed_vertices, axis=-1)
            vertex_colors = dist_to_rgb(vertex_distance, min_dist=0.0, max_dist=3.0)

            for view_id in self.vis_view_ids:
                if self.args.input_image_type == 'stereo_images':
                    if view_id >= self.data['stereo_images'][0].shape[0]:
                        continue
                    input_image = to_numpy(self.data['stereo_images'][0][view_id].permute(1,2,0))
                    camera_intrinsics = to_numpy(self.data['stereo_camera_intrinsics'][0][view_id])
                    camera_extrinsics = to_numpy(self.data['stereo_camera_extrinsics'][0][view_id])
                    radial_distortion = to_numpy(self.data['stereo_camera_distortions'][0][view_id])
                elif self.args.input_image_type == 'color_images':
                    if view_id >= self.data['color_images'][0].shape[0]:
                        continue
                    input_image = to_numpy(self.data['color_images'][0][view_id].permute(1,2,0))
                    camera_intrinsics = to_numpy(self.data['color_camera_intrinsics'][0][view_id])
                    camera_extrinsics = to_numpy(self.data['color_camera_extrinsics'][0][view_id])
                    radial_distortion = to_numpy(self.data['color_camera_distortions'][0][view_id]) 
                else:
                    raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

                if mode == 'train':
                    input_image = self.dataset_train.denormalize_image(input_image)
                else:
                    input_image = self.dataset_val.denormalize_image(input_image)
                input_image = (255*input_image).astype(np.uint8)

                camera_args = {
                    'camera_intrinsics': camera_intrinsics,
                    'camera_extrinsics': camera_extrinsics,
                    'radial_distortion': radial_distortion,
                    'frustum': {'near': 0.01, 'far': 3000.0},
                    'image_size': input_image.shape[:2]
                }

                target_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=None, **camera_args)
                reconstruction_rendering = render_mesh(vertices=reconstructed_vertices, faces=faces, vertex_colors=None, **camera_args)
                target_error_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=vertex_colors, **camera_args)
                visualization = np.hstack((input_image, target_rendering, reconstruction_rendering, target_error_rendering)).transpose(2,0,1)
                self.tb_logger.add_image('%s/view_id_%02d' % (mode.capitalize(), view_id), visualization, self.global_step)

    def _points_reconstruction_loss(self):
        '''
        Vertex-to-vertex difference between the target meshes and the reconstructed points. 
        '''

        return self.points_loss_function(self.global_points, self.target_vertices)

    def _points2surface_loss(self, no_robustifier=False):
        '''
        Point-to-surface difference between the target scans and the surface spanned by the reconstructed points. 
        The point-to-surface loss computes for every vertex in the target scan the distance to the closest point in the 
        surface of the mesh defined by the reconstructed points. 
        '''

        if 'v_scan' not in self.data:
            print("No scan vertices available")
            return 0.0
        scan_vertices = self.data['v_scan'].to(self.device)
        predicted_vertices = self.global_points
        predicted_faces = self.data['f_reg_global'][0].to(self.device)
        if not no_robustifier:
            return self.points2surface_loss_function(scan_vertices, predicted_vertices, predicted_faces)
        else:
            with torch.no_grad():
                distances = compute_s2m_distance(scan_vertices, predicted_vertices, predicted_faces)
                return distances.mean(-1).mean()

    def _edge_regularizer(self):
        return self.edge_loss_function(self.global_points, self.target_vertices)

# -----------------------------------------------------------------------------

def run(config_fname=''):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    parser.print_options()

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set trainer
    trainer = Trainer(args, device)
    trainer.initialize()
    trainer.run()

if __name__ == '__main__':
    run()
    print('Done')

