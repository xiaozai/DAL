import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import scipy.io

def STCDataset():
    return STCDatasetClass().get_sequence_list()


class STCDatasetClass(BaseDataset):
    """STC RGB-D Tracking Benchmark

    Publication:
        Robust fusion of colour and depth data for RGB-D target tracking using adaptive range-invariant depth models and spatio-temporal consistency constraints
        J. Xiao.
        IEEE transaction on cybernetics, 2018
        https://github.com/shine636363/RGBDtracker

    Download the dataset from http:https://github.com/shine636363/RGBDtracker"""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.stc_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        #nz = 8
        ext = 'png'
        start_frame = 1

        anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name,'GT')

        if os.path.exists(str(anno_path)):
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

            end_frame = ground_truth_rect.shape[0]
            ground_truth_rect=ground_truth_rect[:,[0,1,2,3]]

        else:
            #print('ptbdataset, no full groundtruth file, use init file')
            anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
            #print(ground_truth_rect.shape)
            ground_truth_rect=ground_truth_rect.reshape(1,4)

        # frames = ['{base_path}/{sequence_path}/rgb/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        #           sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
        #           for frame_num in range(start_frame, end_frame+1)]

        frames_path = '{}/{}/RGB'.format(self.base_path, sequence_path)
        rgb_frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(ext)]
        rgb_frame_list.sort(key=lambda f: int(f[0:-4]))
        rgb_frame_list = [os.path.join(frames_path, frame) for frame in rgb_frame_list]
        #print('ptbdataset, rgb_frame_list[0] %s' % rgb_frame_list[0])

        depth_frames_path='{}/{}/Depth'.format(self.base_path, sequence_path)
        depth_frame_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith(ext)]
        depth_frame_list.sort(key=lambda f: int(f[0:-4]))
        depth_frame_list = [os.path.join(depth_frames_path, frame) for frame in depth_frame_list]

        if len(ground_truth_rect)==0:
            ground_truth_rect=np.zeros((len(rgb_frame_list),4))

        return Sequence(sequence_name, rgb_frame_list, ground_truth_rect, depth_frame_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if True:
            sequence_list= [
                            'athlete_move',
                            'athlete_static',
                            'backpack_move',
                            'backpack_static',
                            'bag_move',
                            'bag_static',
                            'bin_move',
                            'bin_static',
                            'blanket_move',
                            'blanket_static',
                            'body_move',
                            'body_static',
                            'book_move',
                            'book_static',
                            'cap_move',
                            'cap_static',
                            'doll_move',
                            'doll_static',
                            'face_move',
                            'face_static',
                            'funnel_move',
                            'funnel_static',
                            'gloves_move',
                            'gloves_static',
                            'scarf_move',
                            'scarf_static',
                            'shoe_move',
                            'shoe_static',
                            'toytank_move',
                            'toytank_static',
                            'trolley_move',
                            'trolley_static',
                            'tube_move',
                            'tube_static',
                            'umbrella_move',
                            'umbrella_static'
                            ]



        return sequence_list
