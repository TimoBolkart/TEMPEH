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
import sys
import shutil
import argparse
import numpy as np

def get_date_string():
    from datetime import datetime
    mydate = datetime.now()
    return '%s%02d__%02d-%02d-%02d' % (mydate.strftime("%B"), mydate.day, mydate.hour, mydate.minute, mydate.second)

def execute_locally(test_config_fname):
    from tester.global_tester import run
    run(config_fname=test_config_fname)
    
def run(model_run_dir, data_list_fname, image_directory, calibration_directory, image_file_ext, out_dir):
    if not os.path.exists(model_run_dir):
        print('Model directory not found - %s' % model_run_dir)
        return
    if not os.path.exists(data_list_fname):
        print('Data list not found - %s' % data_list_fname)
        return
    if not os.path.exists(image_directory):
        print('Input iamge directory not found - %s' % image_directory)
        return
    if not os.path.exists(calibration_directory):
        print('Camera calibration directory not found - %s' % calibration_directory)
        return

    os.makedirs(out_dir, exist_ok=True)                
    output_renderings = False

    test_config_fname = os.path.join(out_dir, get_date_string() + '.npz')
    np.savez(test_config_fname, run_dir=model_run_dir, out_dir=out_dir, 
            data_list_fname=data_list_fname, image_directory=image_directory, 
            calibration_directory=calibration_directory, image_file_ext=image_file_ext, 
            output_renderings=output_renderings)

    execute_locally(test_config_fname)

def main():
    parser = argparse.ArgumentParser(description='Instant Multi-View Head Capture through Learnable Registration (TEMPEH)')

    parser.add_argument('--coarse_model_run_dir', default='./runs/coarse/coarse__TEMPEH_final', help='Path of the TEMPEH coarse model directory')
    parser.add_argument('--data_list_fname', default='./data/test_data_subset/paper_test_frames.json', help='Path of the data list file')
    parser.add_argument('--image_directory', default='./data/test_data_subset/test_subset_images_4', help='Path to multi-view image directory')
    parser.add_argument('--calibration_directory', default='./data/test_data_subset/test_subset_calibrations', help='Path to the directory with the camera calibrations')
    parser.add_argument('--image_file_ext', default='png', help='File extension of the multi-view images')
    parser.add_argument('--out_dir', default='./results/FaMoS_test_subset/coarse__TEMPEH_final', help='Output directory for the predicted meshes')

    args = parser.parse_args()
    model_run_dir=args.coarse_model_run_dir
    data_list_fname=args.data_list_fname
    image_directory=args.image_directory
    calibration_directory=args.calibration_directory
    image_file_ext=args.image_file_ext
    out_dir=args.out_dir
    sys.argv = [sys.argv[0]] # Clear the comman-line arguments as the training and testing uses argparse to process the parameters

    run(model_run_dir=model_run_dir, data_list_fname=data_list_fname, image_directory=image_directory, 
        calibration_directory=calibration_directory, image_file_ext=image_file_ext, out_dir=out_dir)

if __name__ == '__main__':
    main()
    print('Done')