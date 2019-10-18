"""Extract joints annotations and match with nuScenes ground truths
"""

import os
import sys
import time
import json
import logging
from collections import defaultdict
import datetime
import math

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

from ..utils import get_iou_matches, append_cluster, select_categories, project_3d
from ..network.process import preprocess_pifpaf, preprocess_monoloco


class PreprocessNuscenes:

    AV_W = 0.68
    AV_L = 0.75
    AV_H = 1.72
    WLH_STD = 0.1

    """
    Preprocess Nuscenes dataset
    """
    CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list)))
              }
    dic_names = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, dir_nuscenes, dataset, iou_min):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.iou_min = iou_min
        self.dir_ann = dir_ann
        dir_out = os.path.join('data', 'arrays')
        assert os.path.exists(dir_nuscenes), "Nuscenes directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-' + dataset + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + dataset + '-' + now_time + '.json')

        self.nusc, self.scenes, self.split_train, self.split_val = factory(dataset, dir_nuscenes)

    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes = cnt_samples = cnt_sd = cnt_ann = 0
        start = time.time()
        for ii, scene in enumerate(self.scenes):
            end_scene = time.time()
            current_token = scene['first_sample_token']
            cnt_scenes += 1
            time_left = str((end_scene - start_scene) / 60 * (len(self.scenes) - ii))[:4] if ii != 0 else "NaN"

            sys.stdout.write('\r' + 'Elaborating scene {}, remaining time {} minutes'
                             .format(cnt_scenes, time_left) + '\t\n')
            start_scene = time.time()
            if scene['name'] in self.split_train:
                phase = 'train'
            elif scene['name'] in self.split_val:
                phase = 'val'
            else:
                print("phase name not in training or validation split")
                continue

            while not current_token == "":
                sample_dic = self.nusc.get('sample', current_token)
                cnt_samples += 1

                # Extract all the sample_data tokens for each sample
                for cam in self.CAMERAS:
                    sd_token = sample_dic['data'][cam]
                    cnt_sd += 1

                    # Extract all the annotations of the person
                    name, boxes_gt, boxes_3d, ys, kk = self.extract_from_token(sd_token)

                    # Run IoU with pifpaf detections and save
                    path_pif = os.path.join(self.dir_ann, name + '.pifpaf.json')
                    exists = os.path.isfile(path_pif)

                    if exists:
                        with open(path_pif, 'r') as file:
                            annotations = json.load(file)
                            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1600, 900))
                    else:
                        continue

                    if keypoints:
                        inputs = preprocess_monoloco(keypoints, kk).tolist()

                        matches = get_iou_matches(boxes, boxes_gt, self.iou_min)
                        for (idx, idx_gt) in matches:
                            self.dic_jo[phase]['kps'].append(keypoints[idx])
                            self.dic_jo[phase]['X'].append(inputs[idx])
                            self.dic_jo[phase]['Y'].append(ys[idx_gt])
                            self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                            self.dic_jo[phase]['boxes_3d'].append(boxes_3d[idx_gt])
                            self.dic_jo[phase]['K'].append(kk)
                            append_cluster(self.dic_jo, phase, inputs[idx], ys[idx_gt], keypoints[idx])
                            cnt_ann += 1
                            sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')

                current_token = sample_dic['next']

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} annotations for {} samples in {} scenes. Total time: {:.1f} minutes"
              .format(cnt_ann, cnt_samples, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))

    def extract_from_token(self, sd_token):

        boxes_gt = []
        ys = []
        boxes_3d = []
        path_im, boxes_obj, kk = self.nusc.get_sample_data(sd_token, box_vis_level=1)  # At least one corner
        kk = kk.tolist()
        name = os.path.basename(path_im)

        # if name == 'n015-2018-08-01-16-32-59+0800__CAM_FRONT__1533112709162460.jpg':
        # if name == 'n015-2018-07-18-11-50-34+0800__CAM_BACK_RIGHT__1531885876377893.jpg':
        #     aa = 5

        for box_obj in boxes_obj:
            if box_obj.name[:6] != 'animal':
                general_name = box_obj.name.split('.')[0] + '.' + box_obj.name.split('.')[1]
            else:
                general_name = 'animal'
            if general_name in select_categories('all'):
                box = project_3d(box_obj, kk)
                dd = np.linalg.norm(box_obj.center)
                yaw = quaternion_yaw(box_obj.orientation)
                sin, cos = correct_angle(yaw, box_obj)
                boxes_gt.append(box)
                xyz = list(box_obj.center)
                wlh = list((box_obj.wlh - np.array([self.AV_W, self.AV_L, self.AV_H])) / self.WLH_STD)
                output = [dd] + xyz + wlh + [sin, cos, yaw]
                box_3d = box_obj.center.tolist() + box_obj.wlh.tolist()
                ys.append(output)
                boxes_3d.append(box_3d)
                self.dic_names[name]['boxes'].append(box)
                self.dic_names[name]['Y'].append(output)
                self.dic_names[name]['K'] = kk

        return name, boxes_gt, boxes_3d, ys, kk


def factory(dataset, dir_nuscenes):
    """Define dataset type and split training and validation"""

    assert dataset in ['nuscenes', 'nuscenes_mini', 'nuscenes_teaser']
    if dataset == 'nuscenes_mini':
        version = 'v1.0-mini'
    else:
        version = 'v1.0-trainval'

    nusc = NuScenes(version=version, dataroot=dir_nuscenes, verbose=True)
    scenes = nusc.scene

    if dataset == 'nuscenes_teaser':
        with open("splits/nuscenes_teaser_scenes.txt", "r") as file:
            teaser_scenes = file.read().splitlines()
        scenes = [scene for scene in scenes if scene['token'] in teaser_scenes]
        with open("splits/split_nuscenes_teaser.json", "r") as file:
            dic_split = json.load(file)
        split_train = [scene['name'] for scene in scenes if scene['token'] in dic_split['train']]
        split_val = [scene['name'] for scene in scenes if scene['token'] in dic_split['val']]
    else:
        split_scenes = splits.create_splits_scenes()
        split_train, split_val = split_scenes['train'], split_scenes['val']

    return nusc, scenes, split_train, split_val


def quaternion_yaw(q: Quaternion, in_image_frame: bool = True) -> float:
    if in_image_frame:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
    return float(yaw)


def correct_angle(yaw, box_obj):

    correction = math.atan2(box_obj.center[0], box_obj.center[2])
    yaw = yaw - correction
    if yaw > np.pi:
        yaw -= 2 * np.pi
    elif yaw < -np.pi:
        yaw += 2 * np.pi
    assert -2 * np.pi <= yaw <= 2 * np.pi
    return math.sin(yaw), math.cos(yaw)


def extract_box_average(boxes_3d):
    boxes_np = np.array(boxes_3d)
    means = np.mean(boxes_np[:, 3:], axis=0)
    stds = np.std(boxes_np[:, 3:], axis=0)
    print(means)
    print(stds)




# def get_jean_yaw(box_obj):
#     b_corners = box_obj.bottom_corners()
#     center = box_obj.center
#     back_point = [(b_corners[0, 2] + b_corners[0, 3]) / 2, (b_corners[2, 2] + b_corners[2, 3]) / 2]
#
#     x = b_corners[0, :] - back_point[0]
#     y = b_corners[2, :] - back_point[1]
#
#     angle = math.atan2((x[0] + x[1]) / 2, (y[0] + y[1]) / 2) * 180 / 3.14
#     angle = (angle + 360) % 360
#     correction = math.atan2(center[0], center[2]) * 180 / 3.14
#     return angle, correction
