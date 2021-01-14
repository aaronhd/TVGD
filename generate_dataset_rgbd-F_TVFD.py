# coding: utf-8
import datetime
import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp_force
from sklearn import preprocessing
import cv2
import provider
from copy import  deepcopy

# This script is used for TVGD paper website and the dataset.


# Raw data path
RAW_DATA_DIR = '/media/aarons/hdd_2/ggcnn_dataset/cornell'  # Cornell dataset path
RAW_DATA_DIR_tactile = '/media/aarons/hdd_2/corrnel_grasp_dataset/tactile'  # force data path

# Output path
OUTPUT_DIR = '/media/aarons/hdd_2/corrnel_grasp_dataset/hdf5/RGB-F'


#This is RGB + depth
DATASET_NAME = 'dataset'
OUTPUT_IMG_SIZE = (336, 336)  # using for training u-net
RANDOM_ROTATIONS = 16
RANDOM_ZOOM = True

TRAIN_SPLIT = 0.8
# OR specify which images are in the test set.
TEST_IMAGES = None
VISUALISE_ONLY = True

# File name patterns for the different file types.  _ % '<image_id>'
_rgb_pattern = os.path.join(RAW_DATA_DIR, 'pcd%sr.png')
_pcd_pattern = os.path.join(RAW_DATA_DIR, 'pcd%s.txt')
_pos_grasp_pattern = os.path.join(RAW_DATA_DIR_tactile, 'pcd%scpos_force.txt')
_neg_grasp_pattern = os.path.join(RAW_DATA_DIR, 'pcd%scneg.txt')


def get_image_ids():
    # Get all the input files, extract the numbers.

    rgb_images = glob.glob(_rgb_pattern % '*') 
    rgb_images.sort()
    return [r[-9:-5] for r in rgb_images]


if __name__ == '__main__':
    MAX_WIDTH = 0
    MIN_WIDTH = 335
    # Create the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Label the output file with the date/time it was created
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    outfile_name = os.path.join(OUTPUT_DIR, '%s_%s.hdf5' % (DATASET_NAME, dt))

    fields = [
        'img_id',
        'rgb',
        'grey',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width',
        'force_img'
    ]

    # Empty datatset.
    dataset = {
        'test':  dict([(f, []) for f in fields]),
        'train': dict([(f, []) for f in fields])
    }

    for img_id in get_image_ids():
        print('Processing: %s' % img_id)
        ds_output = 'train'
        if TEST_IMAGES:
            if int(img_id) in TEST_IMAGES:
                print("This image is in TEST_IMAGES")
                ds_output = 'test'
        elif np.random.rand() > TRAIN_SPLIT:
            ds_output = 'test'

        ds = dataset[ds_output]

        # Load the image
        rgb_img_base = Image(io.imread(_rgb_pattern % img_id))
        depth_img_base = DepthImage.from_pcd(_pcd_pattern % img_id, (480, 640))
        depth_img_base.inpaint()

        # Load Grasps.
        bounding_boxes_base = grasp_force.BoundingBoxes.load_from_file_force(_pos_grasp_pattern % img_id)
        center = bounding_boxes_base.center

        for i in range(RANDOM_ROTATIONS):
            angle = np.random.random() * 2 * np.pi - np.pi
            rgb = rgb_img_base.rotated(angle, center)
            depth = depth_img_base.rotated(angle, center)
            bbs = bounding_boxes_base.copy()

            bbs.rotate(angle, center)

            left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
            right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

            top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
            bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

            rgb.crop((top, left), (bottom, right))
            depth.crop((top, left), (bottom, right))

            bbs.offset((-top, -left))

            if RANDOM_ZOOM:
                zoom_factor = np.random.uniform(0.4, 1.0)
                rgb.zoom(zoom_factor)
                depth.zoom(zoom_factor)
                bbs.zoom(zoom_factor, (OUTPUT_IMG_SIZE[0]//2, OUTPUT_IMG_SIZE[1]//2))

            depth.normalise()   
            rgb_raw_crop = rgb.img
            rgb_normalise = np.zeros(rgb.img.shape, dtype=np.float32)
            cv2.normalize(rgb_raw_crop, rgb_normalise, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            grey_raw_crop = cv2.cvtColor(rgb.img, cv2.COLOR_RGB2GRAY)
            grey_normalise = (grey_raw_crop - grey_raw_crop.min())/np.float32(grey_raw_crop.max() - grey_raw_crop.min())
            pos_img, ang_img, width_img, force_img = bbs.draw(depth.shape)
            if MAX_WIDTH < np.max(width_img):
                MAX_WIDTH = np.max(width_img)
            print('MINMAX_WIDTH', MIN_WIDTH, MAX_WIDTH)
            light_depth = depth.img.astype('float16')


            if VISUALISE_ONLY:
                f = plt.figure()

                ax = f.add_subplot(2, 2, 1)
                ax.axis('off')
                # ax.set_xlabel('v')
                ax.set_title('Raw_image', fontsize=14)
                rgb.show(ax)
                bbs.show(ax)

                ax = f.add_subplot(2, 2, 2)
                ax.set_title('Depth_Image', fontsize=14)
                ax.axis('off')
                depth.show(ax)
                bbs.show(ax)

                ax = f.add_subplot(2, 2, 3)
                ax.set_title('Pose_label', fontsize=14)
                ax.axis('off')
                ax.imshow(np.uint8(pos_img * 255.0))
                bbs.show(ax)

                ax = f.add_subplot(2, 2, 4)
                ax.set_title('Force_label', fontsize=14)
                ax.axis('off')
                ax.imshow(np.uint8(force_img * 255.0))
                bbs.show(ax)

                plt.show()
                continue
            ds['img_id'].append(int(img_id))
            ds['rgb'].append(rgb_normalise.astype('float16'))
            ds['grey'].append(grey_normalise.astype('float16'))
            ds['depth_inpainted'].append(light_depth)
            ds['grasp_points_img'].append(pos_img)
            ds['angle_img'].append(ang_img)
            ds['grasp_width'].append(width_img)
            ds['force_img'].append(force_img)


    # Save the output.
    if not VISUALISE_ONLY:
        with h5py.File(outfile_name, 'w') as f:
            for tt_name in dataset:
                # tt_name: train test
                for ds_name in dataset[tt_name]:
                    #  field name
                    f.create_dataset('%s/%s' % (tt_name, ds_name), data=np.array(dataset[tt_name][ds_name]))