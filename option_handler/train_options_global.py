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

import argparse
from option_handler.base_train_options import BaseTrainOptions, json_dict

PRE_TRAINING_POINT_MASK_WEIGHTS = {
    'w_point_face': 1.0,
    'w_point_ears': 1.0,
    'w_point_eyeballs': 1.0,
    'w_point_eye_region': 1.0,
    'w_point_lips': 1.0,
    'w_point_neck': 1.0,
    'w_point_nostrils': 1.0,
    'w_point_scalp': 1.0,
    'w_point_boundary': 1.0,
}

S2M_POINT_MASK_WEIGHTS = {
    'w_point_face': 0.0,
    'w_point_ears': 0.0,
    'w_point_eyeballs': 1.0,
    'w_point_eye_region': 0.0,
    'w_point_lips': 0.0,
    'w_point_neck': 0.0,
    'w_point_nostrils': 0.0,
    'w_point_scalp': 0.0,
    'w_point_boundary': 0.0,
}

EDGE_MASK_WEIGHTS = {
    'w_edge_face': 2.0,
    'w_edge_ears': 3.0,
    'w_edge_eyeballs': 50.0,
    'w_edge_eye_region': 10.0,
    'w_edge_lips': 3.0,
    'w_edge_neck': 1.0,
    'w_edge_nostrils': 10.0,
    'w_edge_scalp': 1.0,
    'w_edge_boundary': 10.0,
}

class TrainOptions(BaseTrainOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseTrainOptions.initialize(self)
        self.isTrain = True
        return self.parser

    def initialize_extra(self):
        self.add_arg( cate='base', abbr='cf',  name='config-filename', type=str, default='/is/ps3/tbolkart/misc_repo/TEMPEH_dev/runs/coarse/coarse__TEMPEH__May07__11-19-35/config.json')
        self.add_arg( cate='base', abbr='md',  name='model-directory', type=str, default='./runs/coarse')
        self.add_arg( cate='base', abbr='eid', name='experiment-id', type=str, default='TEMPEH')
        # data
        self.add_arg( cate='data', abbr='num-spnts', name='number-sample-points', type=list, default=[5023])
        self.add_arg( cate='data', abbr='templ-name', name='template-fname', type=str, default='./data/template/sampling_template.obj')
        self.add_arg( cate='data', abbr='vmask-name', name='vertex-mask-fname', type=str, default='./data/template/vertex_masks2.npz')
        self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='stereo_images') 
        # self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='color_images') 
        self.add_arg( cate='data', abbr='b-sigma', name='brightness-sigma', type=float, default=0.33) 
        self.add_arg( cate='data', abbr='v_scan', name='scan-vertex-count', type=str, default=20000)


        self.add_arg( cate='data', abbr='tdl', name='train-data-list-fname', type=str, default='./data/training_data/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json')
        self.add_arg( cate='data', abbr='vdl', name='val-data-list-fname', type=str, default='./data/training_data/eight_subj__all_seq_frames_per_seq_5_val.json')

        self.add_arg( cate='data', abbr='data-dir', name='dataset-directory', type=str, default='')
        self.add_arg( cate='data', abbr='scan-dir', name='scan-directory', type=str, default='./data/training_data/sampled_scan_points')
        self.add_arg( cate='data', abbr='reg-dir', name='processed-directory', type=str, default='./data/training_data/registrations')

        self.add_arg( cate='data', abbr='image-dir', name='image-directory', type=str, default='./data/training_data/downsampled_images_4')
        self.add_arg( cate='data', abbr='image-ext', name='image-file-ext', type=str, default='png')
        self.add_arg( cate='data', abbr='calib-dir', name='calibration-directory', type=str, default='./data/training_data/calibrations')

        # train
        self.add_arg( cate='train', abbr='niter',  name='num-iterations', type=int, default=argparse.SUPPRESS)  # global
        self.add_arg( cate='train', abbr='print-freq',  name='print-frequency', type=int, default=100)
        self.add_arg( cate='train', abbr='vis-freq',  name='visualize-frequency', type=int, default=5000)
        self.add_arg( cate='train', abbr='val-freq',  name='validate-frequency', type=int, default=5000)
        self.add_arg( cate='train', abbr='save-freq',  name='save-frequency', type=int, default=50000)

        # model
        self.add_arg( cate='model', abbr='gfeature-fusion', name='global-feature-fusion', type=str, default='mean_var') 
        # self.add_arg( cate='model', abbr='gtrafo', name='global-spatial-transformer', type=str, default='none') # baseline
        self.add_arg( cate='model', abbr='gtrafo', name='global-spatial-transformer', type=str, default='rigid_transformer') # TEMPEH


        self.add_arg( cate='model', abbr='gtrafo-dim', name='global-spatial-transformer-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='gtrafo-levels', name='global-transformer-levels', type=int, default=2)
        self.add_arg( cate='model', abbr='gtrafo-sfactor', name='global-transformer-scale-factor', type=float, default=0.4)
        self.add_arg( cate='model', abbr='gtrafo-pooling', name='global-transformer-use-pooling', type=bool, default=True)

        self.add_arg( cate='model', abbr='sviews', name='sample-views', type=bool, default=True)   
        self.add_arg( cate='model', abbr='min-views', name='minimum-sample-views', type=int, default=8)
        self.add_arg( cate='model', abbr='irf', name='image-resize-factor', type=int, default=8) 
        self.add_arg( cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=8) 
        self.add_arg( cate='model', abbr='feat-arch', name='feature-arch', type=str, default='uresnet2')

        # self.add_arg( cate='model', abbr='global-arch', name='global-arch', type=str, default='v2v') # baseline
        self.add_arg( cate='model', abbr='global-arch', name='global-arch', type=str, default='v2v2') # TEMPEH
        self.add_arg( cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')

        # volumetric sparse point net
        self.add_arg( cate='model', abbr='gvd', name='global-voxel-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='gvi', name='global-voxel-inc', type=float, default=25.0)
        self.add_arg( cate='model', abbr='go', name='global-origin', type=list, default=[0.0, -50.0, 0.0])
        self.add_arg( cate='model', abbr='nm', name='norm', type=str, default="bn")

        # loss functions
        self.add_arg( cate='model', abbr='p2p', name='point2point-loss-function', type=str, default='mse')

        # loss weights
        self.add_arg( cate='model', abbr='wpointr', name='weight-points-recon', type=float, default=1.0) 
        self.add_arg( cate='model', abbr='wpointm', name='point-mask-weights', type=json_dict, default=argparse.SUPPRESS) # pre-training: PRE_TRAINING_POINT_MASK_WEIGHTS, afterwards: S2M_POINT_MASK_WEIGHTS
        self.add_arg( cate='model', abbr='wp2s', name='weight-points2surface', type=float, default=argparse.SUPPRESS) # pre-training: 0.0, afterwards: 10.0
        self.add_arg( cate='model', abbr='gmo', name='gmo_sigma', type=float, default=10.0)
        self.add_arg( cate='model', abbr='wedge', name='weight-edge-regularizer', type=float, default=argparse.SUPPRESS) # pre-training: 0.0, afterwards: 10.0
        self.add_arg( cate='model', abbr='wedgem', name='edge-mask-weights', type=dict, default=EDGE_MASK_WEIGHTS)

        self.initialize_default_parameters()

    def initialize_default_parameters(self):
        self.default_parameters = {
            'num-iterations': 300000,
            'weight-points2surface': 0.0,
            'weight-edge-regularizer': 0.0,
            'point-mask-weights': PRE_TRAINING_POINT_MASK_WEIGHTS
        }