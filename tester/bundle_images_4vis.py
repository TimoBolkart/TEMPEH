import os
import imageio
import numpy as np
from skimage.transform import resize

img_dir = lambda subject, sequence, frame : os.path.join('/ps/data/Faces/FaMoS', subject, sequence, 'images', frame)

# FaMoS_180703_03354_TA_mouth_extreme_000077_img
# FaMoS_180424_03335_TA_kissing_000046_img
# FaMoS_180703_03354_TA_head_rotation_up_down_000035_img
# FaMoS_180424_03335_TA_smile_closed_000088
# FaMoS_180424_03335_TA_blow_cheeks_000064

# FaMoS_180424_03335_TA_kissing.000046
# FaMoS_180703_03354_TA_head_rotation_up_down.000035
# FaMoS_180710_03361_TA_blow_cheeks.000141
# FaMoS_180720_03318_TA_head_rotation_up_down.000098
# FaMoS_181002_03384_TA_head_rotation_left_right.000168
# FaMoS_181214_50039_TA_kissing.000062
# FaMoS_190116_03367_TA_mouth_extreme.000046
# FaMoS_190206_03444_TA_head_rotation_up_down.000246

# FaMoS_181211_03423_TA_mouth_extreme_000036
# FaMoS_190226_00167_TA_mouth_side_000116
# FaMoS_181002_03384_TA_mouth_extreme_000054

subject = 'FaMoS_180424_03335_TA'
sequence = 'high_smile'
frame = '000032'
contrast_factor = 1.5

# List of teaser figures
sample_list = [
    ['FaMoS_180424_03335_TA', 'blow_cheeks', '000119'],
    ['FaMoS_180424_03335_TA', 'high_smile', '000029'],
    ['FaMoS_180424_03335_TA', 'kissing', '000046'],
    ['FaMoS_180424_03335_TA', 'mouth_extreme', '000038'],
    ['FaMoS_180424_03335_TA', 'head_rotation_left_right', '000207'],
    ['FaMoS_180424_03335_TA', 'head_rotation_up_down', '000047'],
    ['FaMoS_180424_03335_TA', 'head_rotation_up_down', '000210']
    ]

# sample_list = [
#     ['FaMoS_180424_03335_TA', 'kissing', '000038'],
#     ['FaMoS_180424_03335_TA', 'high_smile', '000043'], 
#     ['FaMoS_180703_03354_TA', 'mouth_extreme', '000077'],
#     ['FaMoS_180703_03354_TA', 'high_smile', '000053'],    
#     ['FaMoS_180703_03354_TA', 'head_rotation_left_right', '000144'],
#     ['FaMoS_180705_03359_TA', 'happiness', '000058'],
#     ['FaMoS_180705_03359_TA', 'disgust', '000079'],
#     ['FaMoS_180710_03361_TA', 'smile_closed', '000090'],
#     ['FaMoS_180924_03379_TA', 'mouth_side', '000203'],
#     ['FaMoS_180924_03379_TA', 'high_smile', '000121'], 
#     ['FaMoS_181002_03384_TA', 'jaw', '000080'],
#     ['FaMoS_181002_03384_TA', 'head_rotation_left_right', '000168'],
#     ['FaMoS_181017_03402_TA', 'mouth_middle', '000143'],
#     ['FaMoS_181204_03415_TA', 'high_smile', '000064'],
#     ['FaMoS_181204_03415_TA', 'mouth_extreme', '000084'],
#     ['FaMoS_181211_03423_TA', 'mouth_middle', '000138'],
#     ['FaMoS_181214_50039_TA', 'blow_cheeks', '000121'],
#     ['FaMoS_190116_03367_TA', 'mouth_side', '000216'],
#     ['FaMoS_190206_03444_TA', 'lips_up', '000060'],
#     ['FaMoS_190226_00167_TA', 'mouth_up', '000017'],
#     ['FaMoS_190226_00167_TA', 'mouth_side', '000116'],
#     ]

# sample_list = [
#     ['FaMoS_190227_03467_TA', 'mouth_extreme', '000140'],
#     ['FaMoS_190206_03444_TA', 'head_rotation_left_right', '000130'],
#     ['FaMoS_190206_03444_TA', 'happiness', '000085'],
#     ['FaMoS_190206_03444_TA', 'head_rotation_up_down', '000120'],
#     ['FaMoS_190116_03367_TA', 'high_smile', '000062'],
#     ['FaMoS_190116_03367_TA', 'mouth_extreme', '000046'],
#     ['FaMoS_181214_50039_TA', 'kissing', '000081'],
#     ['FaMoS_181214_50039_TA', 'mouth_up', '000151'],
#     ['FaMoS_181214_50039_TA', 'mouth_open', '000103'],
#     ['FaMoS_181211_03423_TA', 'head_rotation_left_right', '000129'],
#     ['FaMoS_181211_03423_TA', 'head_rotation_up_down', '000056'],
#     ['FaMoS_181211_03423_TA', 'mouth_extreme', '000057'],
#     ['FaMoS_181204_03415_TA', 'mouth_side', '000053'],
#     ['FaMoS_180924_03379_TA', 'mouth_up', '000132'],
#     ['FaMoS_180924_03379_TA', 'high_smile', '000111'],
#     ['FaMoS_180924_03379_TA', 'blow_cheeks', '000051']
#     ]

# for spatial transformer:
# FaMoS_180705_03359_TA_head_rotation_left_right        (max x-diff)
# FaMoS_181211_03423_TA_kissing     (max z-diff)
# FaMoS_190206_03444_TA_head_rotation_left_right  - FaMoS_180705_03359_TA_head_rotation_left_right  
# FaMoS_181017_03402_TA_jaw - FaMoS_181002_03384_TA_head_rotation_left_right

# sample_list = []

# subj_seq_list = ['FaMoS_180424_03335_TA_blow_cheeks', 'FaMoS_180424_03335_TA_high_smile', 'FaMoS_180424_03335_TA_kissing',
#                 'FaMoS_180424_03335_TA_mouth_extreme', 'FaMoS_180424_03335_TA_head_rotation_left_right', 'FaMoS_180424_03335_TA_head_rotation_up_down',
#                 'FaMoS_180703_03354_TA_head_rotation_left_right', 'FaMoS_180705_03359_TA_happiness', 'FaMoS_181002_03384_TA_jaw',
#                 'FaMoS_181017_03402_TA_mouth_middle', 'FaMoS_181214_50039_TA_mouth_up', 'FaMoS_190227_03467_TA_mouth_extreme', 'FaMoS_181211_03423_TA_mouth_extreme',
#                 'FaMoS_180703_03354_TA_head_rotation_up_down', 'FaMoS_181002_03384_TA_head_rotation_left_right']

# Teaser videos
# subj_seq_list = ['FaMoS_180424_03335_TA_blow_cheeks', 'FaMoS_180424_03335_TA_high_smile', 'FaMoS_180424_03335_TA_kissing',
#                 'FaMoS_180424_03335_TA_mouth_extreme', 'FaMoS_180424_03335_TA_head_rotation_left_right', 'FaMoS_180424_03335_TA_head_rotation_up_down']

# Spatial transformer examples
# subj_seq_list = ['FaMoS_180705_03359_TA_head_rotation_left_right', 'FaMoS_181211_03423_TA_kissing', 'FaMoS_190206_03444_TA_head_rotation_left_right',
#                 'FaMoS_181017_03402_TA_jaw', 'FaMoS_181002_03384_TA_head_rotation_left_right']


# subj_seq_list = ['FaMoS_180705_03359_TA_happiness', 'FaMoS_181002_03384_TA_jaw',
#                 'FaMoS_181017_03402_TA_mouth_middle', 'FaMoS_181214_50039_TA_mouth_up', 'FaMoS_190227_03467_TA_mouth_extreme', 'FaMoS_181211_03423_TA_mouth_extreme',
#                 'FaMoS_180703_03354_TA_head_rotation_up_down', 'FaMoS_181002_03384_TA_head_rotation_left_right']

# subj_seq_list = ['FaMoS_180703_03354_TA_mouth_side', 'FaMoS_181002_03384_TA_blow_cheeks', 'FaMoS_181002_03384_TA_jaw', 'FaMoS_181211_03423_TA_kissing', 'FaMoS_190227_03467_TA_high_smile']
subj_seq_list = ['FaMoS_181002_03384_TA_blow_cheeks', 'FaMoS_181002_03384_TA_jaw']


global_local = 'local'
run_id = 'local__vis_normal_filterd_v2_s2m_30.0_reg_0.1_gmo_sigma_10.0_soft_edge_weighting4__November03__11-56-22'
result_dir = '/is/ps3/tbolkart/misc_repo/ToFu_dev/results/test_predictions/%s_sequences/%s' % (global_local, run_id)

out_dir = '/ps/project/famos/FaMoS/bundled_images_2x2'
os.makedirs(out_dir, exist_ok=True)

# calib_fnames = [['25_A', '26_A', '24_A'],
#                 ['25_B', '26_B', '24_B'],
#                 ['28_A', '23_A', '27_A'],
#                 ['28_B', '23_B', '27_B'],
#                 ['30_B', '30_A', '29_A', '29_B']]

calib_fnames = [['25_B', '26_B'],
                ['28_A', '23_A']]

import glob
from utils.utils import get_sub_folder, get_filename
subjects = get_sub_folder(result_dir)
for subject in subjects:
    sequences = get_sub_folder(os.path.join(result_dir, subject))
    for sequence in sorted(sequences):
        if '%s_%s' % (subject, sequence) not in subj_seq_list:
            continue

        fnames = sorted(glob.glob(os.path.join(result_dir, subject, sequence, sequence + '.*.ply')))
        for fname in fnames:
            idx1 = len(sequence)+1
            frame = get_filename(fname)[idx1:idx1+7]

            rows = []
            for calib_row in calib_fnames:
                row = []
                for calib_id in calib_row:
                    image_fname = os.path.join(img_dir(subject, sequence, frame), '%s.%s.%s.bmp' % (sequence, frame, calib_id))
                    if not os.path.exists(image_fname):
                        print('Image not found - %s' % image_fname)
                        break

                    image = np.clip(contrast_factor*(imageio.imread(image_fname, pilmode='RGB').astype(np.float32) / 255.), 0.0, 1.0)
                    row.append(image)
                rows.append(np.hstack(row))

            bundled_image = np.vstack(rows)
            bundled_image = resize(bundled_image, (bundled_image.shape[0]//4, bundled_image.shape[1]//4), anti_aliasing=True)

            bundled_image = (255*bundled_image).astype(np.uint8)

            seq_out_dir = os.path.join(out_dir, subject, sequence)
            os.makedirs(seq_out_dir, exist_ok=True)

            img_fname = os.path.join(seq_out_dir, '%s_%s_%s_img.png' % (subject, sequence, frame))
            imageio.imsave(img_fname, bundled_image)



import pdb; pdb.set_trace()












for sample_id in range(len(sample_list)):
    subject, sequence, frame = sample_list[sample_id]

    # out_dir = os.path.join('/ps/project/famos/FaMoS/bundled_images', subject, sequence)
    out_dir = '/ps/project/famos/FaMoS/bundled_images'
    os.makedirs(out_dir, exist_ok=True)

    # calib_fnames = [['25_B', '26_B', '24_B'],
    #                 ['28_B', '23_B', '27_B'],
    #                 ['30_B', '30_A', '29_A', '29_B']]
    #                 # ['29_A', '29_B', '30_A', '30_B']]

    calib_fnames = [['25_A', '26_A', '24_A'],
                    ['25_B', '26_B', '24_B'],
                    ['28_A', '23_A', '27_A'],
                    ['28_B', '23_B', '27_B'],
                    ['30_B', '30_A', '29_A', '29_B']]

    rows = []
    for calib_row in calib_fnames:
        row = []
        for calib_id in calib_row:
            image_fname = os.path.join(img_dir(subject, sequence, frame), '%s.%s.%s.bmp' % (sequence, frame, calib_id))
            if not os.path.exists(image_fname):
                print('Image not found - %s' % image_fname)
                break

            image = np.clip(contrast_factor*(imageio.imread(image_fname, pilmode='RGB').astype(np.float32) / 255.), 0.0, 1.0)
            row.append(image)
        rows.append(np.hstack(row))

    bundled_image = np.vstack(rows)
    bundled_image = resize(bundled_image, (bundled_image.shape[0]//4, bundled_image.shape[1]//4), anti_aliasing=True)

    bundled_image = (255*bundled_image).astype(np.uint8)

    img_fname = os.path.join(out_dir, '%s_%s_%s_img.png' % (subject, sequence, frame))
    imageio.imsave(img_fname, bundled_image)

