import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

def DepthTrackDataset():
    return DepthTrackDatasetClass().get_sequence_list()


class DepthTrackDatasetClass(BaseDataset):
    """VOT2021 Sequestered RGBD dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_path
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
        sequence_list = os.listdir(self.base_path)
        try:
            sequence_list.remove('list.txt')
        except:
            pass

        return sequence_list
