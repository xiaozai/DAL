import os
import os.path
import torch
import numpy as np
import pandas
import csv
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader,opencv_loader
from ltr.admin.environment import env_settings

import cv2 as cv


class kevinlaiRGBD(BaseDataset):
    """kevin-lai RGB-D Tracking Benchmark

    Publication:
            A Large-Scale Hierarchical Multi-View RGB-D Object Dataset
            Kevin Lai, Liefeng Bo, Xiaofeng Ren, and Dieter Fox
            IEEE International Conference on Robotics and Automation (ICRA), May 2011.

    Download the dataset from http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes/README.txt"""

    def __init__(self, root=None, image_loader=opencv_loader, vid_ids=None, split=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
        """
        root = env_settings().kevin_dir if root is None else root
        print('init kevinlaiRGBD, root is %s'%root)
        super().__init__(root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'kevinlai_list.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def get_name(self):
        return 'kevinlaiRGBD'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self.sequence_list[seq_id].split('/')[-1]
        anno_file = os.path.join(seq_path, "%s.txt"%(seq_name))
        #print('seq_path is %s, anno_file is %s' % (seq_path, anno_file))
        #print(anno_file)
        #gt = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float64, na_filter=False, low_memory=False).values
        gt=np.loadtxt(str(anno_file), delimiter=',', dtype=np.float32)
        gt=gt[:,[0,1,2,3]]
        return torch.Tensor(gt)

    # def _read_target_visible(self, seq_path, anno):
    #     # Read full occlusion and out_of_view
    #     occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
    #     out_of_view_file = os.path.join(seq_path, "out_of_view.txt")
    #
    #     with open(occlusion_file, 'r', newline='') as f:
    #         occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #     with open(out_of_view_file, 'r') as f:
    #         out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #
    #     target_visible = ~occlusion & ~out_of_view & (anno[:,2]>0) & (anno[:,3]>0)
    #
    #     return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('/')[1]
        #vid_id = seq_name.split('-')[1]
        #return os.path.join(self.root, class_name, class_name + '-' + vid_id)
        return os.path.join(self.root, class_name)

    def _get_smallseq_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('/')[1]
        file_name_woinstance=seq_name.split('/')[2][2:]
        #example ['/home/qian2/qian2_old/kevinlai_RGBD/rgbd-scenes/desk/desk_3']
        return os.path.join(self.root, class_name, file_name_woinstance)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_anno(seq_id)
        #target_visible = self._read_target_visible(seq_path, anno)
        #target_visible = (anno[:,2]>0) & (anno[:,3]>0)
        #print(anno[:,2])
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        target_visible = valid

        return  {'bbox': bbox, 'valid': valid, 'visible': target_visible}

        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible = self._read_target_visible(seq_path) & valid.byte()
        # return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # print([seq_path])
        name_smallseq=seq_path.split('/')[-1]
        return os.path.join(seq_path, '{}_{}.png'.format(name_smallseq,frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        #print(self._get_frame_path(seq_path, frame_id))
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_depth_path(self, seq_path, frame_id):
        name_smallseq=seq_path.split('/')[-1]
        return os.path.join(seq_path, '{}_{}_depth.png'.format(name_smallseq,frame_id+1))    # frames start from 1

    def _get_depth(self, seq_path, frame_id):
        #print(self._get_frame_path(seq_path, frame_id))
        return self._read_depth(self._get_depth_path(seq_path, frame_id))

    def _read_depth(self, image_file: str):
        return cv.imread(image_file,cv.IMREAD_ANYDEPTH)/1000.0

    def _get_class(self, seq_path):
        obj_class = seq_path.split('/')[-2]
        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        smallseq_path = self._get_smallseq_path(seq_id)
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(smallseq_path, f_id) for f_id in frame_ids]

        depth_list = [self._get_depth(smallseq_path, f_id) for f_id in frame_ids]
        # print(self._get_depth_path(seq_path,frame_ids[0]))
        # print(self._get_depth(seq_path,frame_ids[0]))

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # if we have depth, merge depth_list to frame_list

        if len(depth_list)>0:
            rgbd_list=[np.concatenate((frame_list[id],np.expand_dims(depth_list[id],-1)),axis=2) for id in range(len(frame_list))]
            frame_list=rgbd_list

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
