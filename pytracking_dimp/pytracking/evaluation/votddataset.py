import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

def VOTDDataset():
    return VOTDDatasetClass().get_sequence_list()


class VOTDDatasetClass(BaseDataset):
    """VOT2019 RGBD dataset

    Download the dataset from http://www.votchallenge.net/vot2019/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.votd_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name

        anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name,'groundtruth')
        #print(anno_path)

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


        frames_path = '{}/{}/color'.format(self.base_path, sequence_path)
        rgb_frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith('jpg')]
        rgb_frame_list.sort(key=lambda f: int(f[0:8]))
        rgb_frame_list = [os.path.join(frames_path, frame) for frame in rgb_frame_list]
        #print('votddataset, rgb_frame_list[0] %s' % rgb_frame_list[0])

        depth_frames_path='{}/{}/depth'.format(self.base_path, sequence_path)
        depth_frame_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith('png')]
        depth_frame_list.sort(key=lambda f: int(f[0:8]))
        depth_frame_list = [os.path.join(depth_frames_path, frame) for frame in depth_frame_list]

        if len(ground_truth_rect)==0:
            ground_truth_rect=np.zeros((len(rgb_frame_list),4))

        return Sequence(sequence_name, rgb_frame_list, ground_truth_rect, depth_frame_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if True:
            sequence_list= [
                            'backpack_blue',
                            'backpack_robotarm_lab_occ',
                            'backpack_room_noocc_1',
                            'bag_outside',
                            'bicycle2_outside',
                            'bicycle_outside',
                            'bottle_box',
                            'bottle_room_noocc_1',
                            'bottle_room_occ_1',
                            'box1_outside',
                            'box_darkroom_noocc_1',
                            'box_darkroom_noocc_10',
                            'box_darkroom_noocc_2',
                            'box_darkroom_noocc_3',
                            'box_darkroom_noocc_4',
                            'box_darkroom_noocc_5',
                            'box_darkroom_noocc_6',
                            'box_darkroom_noocc_7',
                            'box_darkroom_noocc_8',
                            'box_darkroom_noocc_9',
                            'box_humans_room_occ_1',
                            'box_room_noocc_1',
                            'box_room_noocc_2',
                            'box_room_noocc_3',
                            'box_room_noocc_4',
                            'box_room_noocc_5',
                            'box_room_noocc_6',
                            'box_room_noocc_7',
                            'box_room_noocc_8',
                            'box_room_noocc_9',
                            'box_room_occ_1',
                            'box_room_occ_2',
                            'boxes_backpack_room_occ_1',
                            'boxes_humans_room_occ_1',
                            'boxes_office_occ_1',
                            'boxes_room_occ_1',
                            'cart_room_occ_1',
                            'cartman',
                            'cartman_robotarm_lab_noocc',
                            'case',
                            'container_room_noocc_1',
                            'dog_outside',
                            'human_entry_occ_1',
                            'human_entry_occ_2',
                            'humans_corridor_occ_1',
                            'humans_corridor_occ_2_A',
                            'humans_corridor_occ_2_B',
                            'humans_longcorridor_staricase_occ_1',
                            'humans_shirts_room_occ_1_A',
                            'humans_shirts_room_occ_1_B',
                            'jug',
                            'mug_ankara',
                            'mug_gs',
                            'paperpunch',
                            'person_outside',
                            'robot_corridor_noocc_1',
                            'robot_corridor_occ_1',
                            'robot_human_corridor_noocc_1_A',
                            'robot_human_corridor_noocc_1_B',
                            'robot_human_corridor_noocc_2',
                            'robot_human_corridor_noocc_3_A',
                            'robot_human_corridor_noocc_3_B',
                            'robot_lab_occ',
                            'teapot',
                            'tennis_ball',
                            'thermos_office_noocc_1',
                            'thermos_office_occ_1',
                            'toy_office_noocc_1',
                            'toy_office_occ_1',
                            'trashcan_room_occ_1',
                            'trashcans_room_occ_1_A',
                            'trashcans_room_occ_1_B',
                            'trendNetBag_outside',
                            'trendNet_outside',
                            'trophy_outside',
                            'trophy_room_noocc_1',
                            'trophy_room_occ_1',
                            'two_mugs',
                            'two_tennis_balls',
                            'XMG_outside'
                            ]

        return sequence_list
