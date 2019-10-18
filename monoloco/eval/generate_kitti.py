
"""Run monoloco over all the pifpaf joints of KITTI images
and extract and save the annotations in txt files"""


import os
import glob
import shutil
from collections import defaultdict
import math

import numpy as np
import torch

from ..network import MonoLoco
from ..network.process import preprocess_pifpaf
from ..eval.geom_baseline import compute_distance
from ..utils import get_keypoints, pixel_to_camera, xyz_from_distance, get_calibration, open_annotations, split_training
from .stereo_baselines import baselines_association
from .reid_baseline import ReID, get_reid_features


class GenerateKitti:

    METHODS = ['monoloco', 'geometric']

    def __init__(self, model, dir_ann, p_dropout=0.2, n_dropout=0, stereo=True):

        # Load monoloco
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.monoloco = MonoLoco(model=model, device=device, n_dropout=n_dropout, p_dropout=p_dropout)
        self.dir_ann = dir_ann

        # Extract list of pifpaf files in validation images
        dir_gt = os.path.join('data', 'kitti', 'gt')
        self.set_basename = factory_basename(dir_ann, dir_gt)
        self.dir_kk = os.path.join('data', 'kitti', 'calib')

        # Calculate stereo baselines
        self.stereo = stereo
        if stereo:
            self.baselines = ['ml_stereo', 'pose', 'reid']
            self.cnt_disparity = defaultdict(int)
            self.cnt_no_stereo = 0

            # ReID Baseline
            weights_path = 'data/models/reid_model_market.pkl'
            self.reid_net = ReID(weights_path=weights_path, device=device, num_classes=751, height=256, width=128)
            self.dir_images = os.path.join('data', 'kitti', 'images')
            self.dir_images_r = os.path.join('data', 'kitti', 'images_r')

    def run(self):
        """Run Monoloco and save txt files for KITTI evaluation"""

        cnt_ann = cnt_file = cnt_no_file = 0
        dir_out = {key: os.path.join('data', 'kitti', key) for key in self.METHODS}
        for key in self.METHODS:
            make_new_directory(dir_out[key])
            print("\nCreated empty output directory for {} txt files".format(key))

        if self.stereo:
            for key in self.baselines:
                dir_out[key] = os.path.join('data', 'kitti', key)
                make_new_directory(dir_out[key])
                print("Created empty output directory for {}".format(key))
            print("\n")

        # Run monoloco over the list of images
        for basename in self.set_basename:
            path_calib = os.path.join(self.dir_kk, basename + '.txt')
            annotations, kk, tt = factory_file(path_calib, self.dir_ann, basename)
            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1242, 374))
            assert keypoints, "all pifpaf files should have at least one annotation"
            cnt_ann += len(boxes)
            cnt_file += 1

            # Run the network and the geometric baseline
            xyz, bi, yaw, wlh, varss = self.monoloco.forward(keypoints, kk)
            dds_geom = eval_geometric(keypoints, kk, average_y=0.48)

            # Save the file
            all_outputs = [el.detach().cpu() for el in [xyz, bi, yaw, wlh, varss]] + [dds_geom]

            all_params = [kk, tt]
            for key in self.METHODS:
                path_txt = {key: os.path.join(dir_out[key], basename + '.txt')}
                save_txts(path_txt[key], boxes, all_outputs, all_params, mode=key)

            # Correct using stereo disparity and save in different folder
            # if self.stereo:
            #     zzs = self._run_stereo_baselines(basename, boxes, keypoints, zzs, path_calib)
            #     for key in zzs:
            #         path_txt[key] = os.path.join(dir_out[key], basename + '.txt')
            #         save_txts(path_txt[key], boxes, zzs[key], all_params, mode='baseline')

        print("\nSaved in {} txt {} annotations. Not found {} images".format(cnt_file, cnt_ann, cnt_no_file))

        # if self.stereo:
        #     print("STEREO:")
        #     for key in self.baselines:
        #         print("Annotations corrected using {} baseline: {:.1f}%".format(
        #             key, self.cnt_disparity[key] / cnt_ann * 100))
        #     print("Maximum possible stereo associations: {:.1f}%".format(self.cnt_disparity['max'] / cnt_ann * 100))
        #     print("Not found {}/{} stereo files".format(self.cnt_no_stereo, cnt_file))

    def _run_stereo_baselines(self, basename, boxes, keypoints, zzs, path_calib):

        annotations_r, _, _ = factory_file(path_calib, self.dir_ann, basename, mode='right')
        boxes_r, keypoints_r = preprocess_pifpaf(annotations_r, im_size=(1242, 374))

        # Stereo baselines
        if keypoints_r:
            path_image = os.path.join(self.dir_images, basename + '.png')
            path_image_r = os.path.join(self.dir_images_r, basename + '.png')
            reid_features = get_reid_features(self.reid_net, boxes, boxes_r, path_image, path_image_r)
            zzs, cnt = baselines_association(self.baselines, zzs, keypoints, keypoints_r, reid_features)

            for key in cnt:
                self.cnt_disparity[key] += cnt[key]

        else:
            self.cnt_no_stereo += 1
            zzs = {key: zzs for key in self.baselines}
        return zzs


def save_txts(path_txt, all_inputs, all_outputs, all_params, mode='monoloco'):

    assert mode in ('monoloco', 'geometric', 'baseline')
    if mode == 'baseline':
        zzs = all_outputs
    else:
        xyz, bis, yaw, wlh, varss, dds_geom = all_outputs[:]
    uv_boxes = all_inputs[:]
    kk, tt = all_params[:]

    with open(path_txt, "w+") as ff:
        for idx, uv_box in enumerate(uv_boxes):

            xx = float(xyz[idx][0]) + tt[0]
            yy = float(xyz[idx][1]) + tt[1]
            zz = float(xyz[idx][2]) + tt[2]

            if mode == 'geometric':
                try:
                    zz_geom = math.sqrt(float(dds_geom[idx]) ** 2 - xx ** 2 - yy ** 2)
                except ValueError:
                    zz_geom = 0
                cam_0 = [xx, yy, zz_geom]

            else:
                cam_0 = [xx, yy, zz]

            bi = float(bis[idx].cpu())
            conf = 0.5 * uv_box[-1] / bi

            output_list = uv_box[:-1] + [wlh[idx][2], wlh[idx][0], wlh[idx][1]] + cam_0 + [yaw[idx], conf]

            ff.write("%s " % 'Pedestrian')
            ff.write("%i %i %i " % (-1, -1, 10))
            for el in output_list:
                ff.write("%f " % el)

            # add additional uncertainty information
            if mode == 'monoloco':
                ff.write("%f " % bi)
                ff.write("%f " % float(varss[idx]))
                ff.write("%f " % dds_geom[idx])
            ff.write("\n")


def factory_file(path_calib, dir_ann, basename, mode='left'):
    """Choose the annotation and the calibration files. Stereo option with ite = 1"""

    assert mode in ('left', 'right')
    p_left, p_right = get_calibration(path_calib)

    if mode == 'left':
        kk, tt = p_left[:]
        path_ann = os.path.join(dir_ann, basename + '.png.pifpaf.json')

    else:
        kk, tt = p_right[:]
        path_ann = os.path.join(dir_ann + '_right', basename + '.png.pifpaf.json')

    annotations = open_annotations(path_ann)

    return annotations, kk, tt


def eval_geometric(keypoints, kk, average_y=0.48):
    """ Evaluate geometric distance"""

    dds_geom = []

    uv_centers = get_keypoints(keypoints, mode='center')
    uv_shoulders = get_keypoints(keypoints, mode='shoulder')
    uv_hips = get_keypoints(keypoints, mode='hip')

    xy_centers = pixel_to_camera(uv_centers, kk, 1)
    xy_shoulders = pixel_to_camera(uv_shoulders, kk, 1)
    xy_hips = pixel_to_camera(uv_hips, kk, 1)

    for idx, xy_center in enumerate(xy_centers):
        zz = compute_distance(xy_shoulders[idx], xy_hips[idx], average_y)
        xyz_center = np.array([xy_center[0], xy_center[1], zz])
        dd_geom = float(np.linalg.norm(xyz_center))
        dds_geom.append(dd_geom)

    return dds_geom


def make_new_directory(dir_out):
    """Remove the output directory if already exists (avoid residual txt files)"""
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)


def factory_basename(dir_ann, dir_gt):
    """ Return all the basenames in the annotations folder corresponding to validation images"""

    # Extract ground truth validation images
    names_gt = tuple(os.listdir(dir_gt))
    path_train = os.path.join('splits', 'kitti_train.txt')
    path_val = os.path.join('splits', 'kitti_val.txt')
    _, set_val_gt = split_training(names_gt, path_train, path_val)
    set_val_gt = {os.path.basename(x).split('.')[0] for x in set_val_gt}

    # Extract pifpaf files corresponding to validation images
    list_ann = glob.glob(os.path.join(dir_ann, '*.json'))
    set_basename = {os.path.basename(x).split('.')[0] for x in list_ann}
    set_val = set_basename.intersection(set_val_gt)
    assert set_val, " Missing json annotations file to create txt files for KITTI datasets"
    return set_val
