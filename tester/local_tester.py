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
from os.path import join
import shutil
import random
import imageio

import torch
import numpy as np
from utils.utils import print_memory, to_numpy, get_time_string
from utils.mesh_renderer import render_mesh, dist_to_rgb

from tester.base_tester import BaseTester
from option_handler.train_options_local import TrainOptions

from psbody.mesh import Mesh
from utils.mesh_sampling import MeshSampler


class Tester(BaseTester):
    def __init__(self, args, out_dir, data_list_fname, global_registration_root_dir, device, output_renderings=False):
        super().__init__(args, out_dir)
        self.args = args
        self.data_list_fname = data_list_fname
        self.global_registration_root_dir = global_registration_root_dir
        self.device = device
        self.output_renderings = output_renderings
        self.vis_view_ids = None

    def register_mesh_sampler(self):
        mesh_sampler_fname = join(self.directory_output, 'mesh_sampler.npz')
        self.mesh_sampler = MeshSampler()
        self.mesh_sampler.load(mesh_sampler_fname)

    def register_model(self):
        import models.model_aligner.prototypes.model_local_stage as models
        model = models.Model(args=self.args, mesh_sampler=self.mesh_sampler, feature_net=None) 
        model.initialize(init_method='normal')
        model = model.to(self.device)
        self.model = torch.nn.DataParallel(model)

    def register_dataset(self):
        data_list_fname = self.data_list_fname
        dataset_root_dir = self.args.dataset_directory
        image_dir = self.args.image_directory
        calibration_dir = self.args.calibration_directory
        registration_root_dir = self.args.processed_directory
        global_registration_root_dir = self.global_registration_root_dir
        scan_dir = ''
        scan_vertex_count = 0

        image_resize_factor = self.args.image_resize_factor
        load_stereo_images = self.args.input_image_type == 'stereo_images'
        load_color_images = self.args.input_image_type == 'color_images'
        image_file_ext = self.args.image_file_ext

        from datasets.face_align_dataset_mpi import FaceAlignDatasetMPI       
        self.dataset = FaceAlignDatasetMPI( data_list_fname=data_list_fname,
                                            image_dir=image_dir,
                                            scan_dir=scan_dir,
                                            calibration_dir=calibration_dir,
                                            dataset_root_dir=dataset_root_dir,
                                            registration_root_dir=registration_root_dir,
                                            global_registration_root_dir=global_registration_root_dir,
                                            image_resize_factor=image_resize_factor,
                                            mesh_sampler=self.mesh_sampler,
                                            scan_vertex_count=scan_vertex_count, 
                                            load_stereo_images=load_stereo_images,
                                            load_color_images=load_color_images,
                                            image_file_ext=image_file_ext)
        self.dataloader = self.make_data_loader(self.dataset, cuda=True, shuffle=False)

    def feed_data(self, data):
        self.data = data
        self.inputs = {
            'images': data['stereo_images'].to(self.device),
            'camera_intrinsics': data['stereo_camera_intrinsics'].to(self.device),
            'camera_extrinsics': data['stereo_camera_extrinsics'].to(self.device),
            'camera_distortions': data['stereo_camera_distortions'].to(self.device),
            'camera_centers': data['stereo_camera_centers'].to(self.device),
            'global_points': data['v_reg_global'].to(self.device)
        }        


    def forward(self):
        self.points_list = self.model(**self.inputs, random_grid=False)

    def run(self):
        self.set_eval()
        with torch.no_grad():
            for data in self.dataloader:
                self.feed_data(data)
                self.forward()
                self.export_data()

    def export_data(self):
        to_meter_scale_factor = 0.001
        batch_size = self.points_list[-1].shape[0]
        reconstructed_vertices = to_numpy(self.points_list[-1])
        faces = self.mesh_sampler.get_mesh(0).f

        subjects = self.data['subject']
        sequences = self.data['sequence']
        frames = self.data['frame']

        for idx in range(batch_size):
            subject, sequence, frame = subjects[idx], sequences[idx], frames[idx]
            out_sequence_dir = os.path.join(self.out_dir, subject, sequence)
            os.makedirs(out_sequence_dir, exist_ok=True)

            out_fname = os.path.join(out_sequence_dir, '%s.%s.ply' % (sequence, frame))
            Mesh(to_meter_scale_factor*reconstructed_vertices[idx], faces).write_ply(out_fname)

            if self.output_renderings:
                out_rendering_dir = os.path.join(self.rendering_dir, subject, sequence)
                os.makedirs(out_rendering_dir, exist_ok=True)
                out_rendering_fname = os.path.join(out_rendering_dir, '%s.%s.png' % (sequence, frame))

                target_vertices = to_numpy(self.data['v_registration'][idx])
                renderings = self.visualize(reconstructed_vertices[idx], target_vertices, faces)
                imageio.imsave(out_rendering_fname, renderings)

    def visualize(self, reconstructed_vertices, target_vertices, faces):
        if self.vis_view_ids is None:
            num_views = len(self.data['stereo_images'][0])     
            view_ids = np.arange(num_views)
            random.shuffle(view_ids)
            self.vis_view_ids = view_ids[:6]

        vertex_distance = np.linalg.norm(target_vertices-reconstructed_vertices, axis=-1)
        vertex_colors = dist_to_rgb(vertex_distance, min_dist=0.0, max_dist=3.0)

        view_images = []
        for view_id in self.vis_view_ids:
            if view_id >= self.data['stereo_images'][0].shape[0]:
                continue
            input_image = self.data['stereo_images'][0][view_id].permute(1,2,0).numpy()
            input_image = self.dataset.denormalize_image(input_image)
            input_image = (255*input_image).astype(np.uint8)

            camera_args = {
                'camera_intrinsics': self.data['stereo_camera_intrinsics'][0][view_id].numpy(),
                'camera_extrinsics': self.data['stereo_camera_extrinsics'][0][view_id].numpy(),
                'radial_distortion': self.data['stereo_camera_distortions'][0][view_id].numpy(),
                'frustum': {'near': 0.01, 'far': 3000.0},
                'image_size': input_image.shape[:2]
            }

            target_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=None, **camera_args)
            reconstruction_rendering = render_mesh(vertices=reconstructed_vertices, faces=faces, vertex_colors=None, **camera_args)
            target_error_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=vertex_colors, **camera_args)
            visualization = np.hstack((input_image, target_rendering, reconstruction_rendering, target_error_rendering))
            view_images.append(visualization)
        return np.vstack(view_images)

# -----------------------------------------------------------------------------

def run(config_fname=''):
    test_config = np.load(config_fname)
    local_run_dir = str(test_config['local_run_dir'])
    global_registration_root_dir = str(test_config['global_registration_root_dir'])
    out_dir = str(test_config['out_dir'])
    data_list_fname = str(test_config['data_list_fname'])
    output_renderings = bool(test_config['output_renderings'])

    local_config_fname = os.path.join(local_run_dir, 'config.json')
    if not os.path.exists(local_config_fname):
        print('Config file not found - %s' % local_config_fname)
        return

    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(local_config_fname, os.path.join(out_dir, 'config.json'))

    parser = TrainOptions()
    args = parser.parse(config_filename=local_config_fname)
    args.batch_size = 1
    args.scan_vertex_count = 0

    if 'dataset_directory' in test_config:
        args.dataset_directory = str(test_config['dataset_directory'])
    if 'image_directory' in test_config:
        args.image_directory = str(test_config['image_directory'])
    if 'calibration_directory' in test_config:
        args.calibration_directory = str(test_config['calibration_directory'])
    if 'scan_directory' in test_config:
        args.scan_directory = str(test_config['scan_directory'])     
    if 'processed_directory' in test_config:
        args.processed_directory = str(test_config['processed_directory'])
    if 'image_file_ext' in test_config:
        args.image_file_ext = str(test_config['image_file_ext'])

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set trainer
    tester = Tester(args, out_dir, data_list_fname, global_registration_root_dir, device, output_renderings)
    tester.initialize()
    tester.run()

if __name__ == '__main__':
    import sys
    run(sys.argv[2])
    print('Done')
