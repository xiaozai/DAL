import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

def Laval6dDataset():
    return Laval6dDatasetClass().get_sequence_list()


class Laval6dDatasetClass(BaseDataset):
    """Laval 6DOF Tracking Benchmark

    Publication:
        A Framework for Evaluating 6-DOF Object Trackers
        Mathieu Garon, Denis Laurendeau and Jean-François Lalonde
        ECCV, 2018
        http://vision.gel.ulaval.ca/~jflalonde/projects/6dofObjectTracking/index.html

    Download the dataset from the provided link"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.laval_6d_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        #nz = 8
        ext = 'png'
        start_frame = 1

        anno_path = '{}/processed/{}/{}.txt'.format(self.base_path, sequence_name,'groundtruth')
        print(anno_path)

        if os.path.exists(str(anno_path)):
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

            end_frame = ground_truth_rect.shape[0]
            print(ground_truth_rect.shape)
            ground_truth_rect=ground_truth_rect[:,[0,1,2,3]]

        else:
            print('did not find {}'%anno_path)
            # anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
            # try:
            #     ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            # except:
            #     ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
            # print(ground_truth_rect.shape)
            # ground_truth_rect=ground_truth_rect.reshape(1,4)

        frames_path = '{}/processed/{}'.format(self.base_path, sequence_name)
        rgb_frame_list = ['{frames_path}/{frame}.{ext}'.format(frames_path=frames_path, frame=frame_num, ext=ext) for frame_num in range(end_frame)]
        depth_frame_list= ['{frames_path}/{frame}d.{ext}'.format(frames_path=frames_path, frame=frame_num, ext=ext) for frame_num in range(end_frame)]


        if len(ground_truth_rect)==0:
            ground_truth_rect=np.zeros((len(rgb_frame_list),4))

        return Sequence(sequence_name, rgb_frame_list, ground_truth_rect, depth_frame_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        #if 'ValidationSet' in self.base_path:
        sequence_list= [
                        'dragon_interaction_full',
                        'dragon_interaction_hard',
                        'dragon_interaction_rotation',
                        'dragon_interaction_translation',
                        'lego_interaction_full',
                        'lego_interaction_hard',
                        'lego_interaction_rotation',
                        'lego_interaction_translation',
                        'lego_occlusion_0',
                        'lego_occlusion_h_15',
                        'lego_occlusion_h_30',
                        'lego_occlusion_h_45'
                        ]


        #print(self.base_path)
        #print('length of dataset sequence list: %d'%len(sequence_list))

        return sequence_list
