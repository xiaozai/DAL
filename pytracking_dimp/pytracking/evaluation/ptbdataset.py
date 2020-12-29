import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import scipy.io

def PTBDataset():
    return PTBDatasetClass().get_sequence_list()


class PTBDatasetClass(BaseDataset):
    """Princeton RGB-D Tracking Benchmark

    Publication:
        Tracking Revisited Using RGBD Camera
        S. Song, J. Xiao.
        ICCV, 2013
        http://tracking.cs.princeton.edu/eval.php

    Download the dataset from http://tracking.cs.princeton.edu/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.ptb_path
        self.sequence_list = self._get_sequence_list()

        #shall we use rgb/depth registration and synchronization file
        self.use_regis_synch_file=False
        if self.use_regis_synch_file:
            self.regis_synch_dir=os.path.join(os.path.split(self.base_path)[0],'Reg_andS_Sync')

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        #nz = 8
        ext = 'png'
        start_frame = 1

        anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name,sequence_name)

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

        frames_path = '{}/{}/rgb'.format(self.base_path, sequence_path)
        rgb_frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(ext)]
        #print(rgb_frame_list[0].split('-')[2][0:-4])
        rgb_frame_list.sort(key=lambda f: int(f.split('-')[2][0:-4]))
        rgb_frame_list = [os.path.join(frames_path, frame) for frame in rgb_frame_list]
        #print('ptbdataset, rgb_frame_list[0] %s' % rgb_frame_list[0])

        depth_frames_path='{}/{}/depth'.format(self.base_path, sequence_path)
        depth_frame_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith(ext)]
        depth_frame_list.sort(key=lambda f: int(f.split('-')[2][0:-4]))
        depth_frame_list = [os.path.join(depth_frames_path, frame) for frame in depth_frame_list]

        if len(ground_truth_rect)==0:
            ground_truth_rect=np.zeros((len(rgb_frame_list),4))



        #print([rgb_frame_list[0], depth_frame_list[0]])

        if self.use_regis_synch_file:
            seq_both_list=[seq for seq in os.listdir(self.regis_synch_dir+'/both_sync_and_reg') if seq==sequence_name]
            seq_reg_list =[seq for seq in os.listdir(self.regis_synch_dir+'/only_reg') if seq==sequence_name]
            seq_syn_list =[seq for seq in os.listdir(self.regis_synch_dir+'/only_sync') if seq==sequence_name]
            #print(len(seq_both_list), len(seq_reg_list), len(seq_syn_list))
            if len(seq_both_list)>0:
                mat=scipy.io.loadmat(self.regis_synch_dir+'/both_sync_and_reg/'+sequence_name+'/FrameID_sync.mat')
                ids_sync=mat['FrameID_sync']
                ids_sync=list(ids_sync.flatten())
                name_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith(ext)]
                name_list.sort(key=lambda f: int(f.split('-')[2][0:-4]))
                reordered_name_list=[name_list[id-1] for id in ids_sync]
                depth_frame_list = [os.path.join(self.regis_synch_dir+'/both_sync_and_reg/'+sequence_name+'/registered_depth', frame) for frame in reordered_name_list]

            if len(seq_reg_list)>0:
                depth_frame_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith(ext)]
                depth_frame_list.sort(key=lambda f: int(f.split('-')[2][0:-4]))
                only_reg_depth_dir=self.regis_synch_dir+'/only_reg/'+sequence_name+'/registered_depth'
                depth_frame_list = [os.path.join(only_reg_depth_dir, frame) for frame in depth_frame_list]
                # depth_frame_list = []
                # for frame in depth_frame_list:
                #     print(frame)
                #     if os.path.exists(os.path.join(only_reg_depth_dir, frame)):
                #         depth_frame_list.append(os.path.join(only_reg_depth_dir, frame))
                #     else:
                #         depth_frame_list.append(os.path.join(depth_frames_path, frame))

            if len(seq_syn_list)>0 and sequence_name != 'bear_back':
                mat=scipy.io.loadmat(self.regis_synch_dir+'/only_sync/'+sequence_name+'/FrameID_sync.mat')
                ids_sync=mat['FrameID_sync']
                ids_sync=list(ids_sync.flatten())
                name_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith(ext)]
                name_list.sort(key=lambda f: int(f.split('-')[2][0:-4]))
                reordered_name_list=[name_list[id-1] for id in ids_sync]
                depth_frame_list = [os.path.join(depth_frames_path, frame) for frame in reordered_name_list]



        return Sequence(sequence_name, rgb_frame_list, ground_truth_rect, depth_frame_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        if 'ValidationSet' in self.base_path:
            sequence_list= [
                            'bear_front',
                            'child_no1',
                            'face_occ5',
                            'new_ex_occ4',
                            'zcup_move_1'
                            ]

        if 'EvaluationSet' in self.base_path:
            sequence_list= [
                            # 'bear_front',
                            # 'child_no1',
                            # 'face_occ5',
                            # 'new_ex_occ4',
                            # 'zcup_move_1'
                            #evaluation list
                            'bag1',
                            'basketball1',
                            'basketball2',
                            'basketball2.2',
                            'basketballnew',
                            'bdog_occ2',
                            'bear_back',
                            'bear_change',
                            'bird1.1_no',
                            'bird3.1_no',
                            'book_move1',
                            'book_turn',
                            'book_turn2',
                            'box_no_occ',
                            'br_occ_0',
                            'br_occ1',
                            'br_occ_turn0',
                            'cafe_occ1',
                            'cc_occ1',
                            'cf_difficult',
                            'cf_no_occ',
                            'cf_occ2',
                            'cf_occ3',
                            'child_no2',
                            'computerbar1',
                            'computerBar2',
                            'cup_book',
                            'dog_no_1',
                            'dog_occ_2',
                            'dog_occ_3',
                            'express1_occ',
                            'express2_occ',
                            'express3_static_occ',
                            'face_move1',
                            'face_occ2',
                            'face_occ3',
                            'face_turn2',
                            'flower_red_occ',
                            'gre_book',
                            'hand_no_occ',
                            'hand_occ',
                            'library2.1_occ',
                            'library2.2_occ',
                            'mouse_no1',
                            'new_ex_no_occ',
                            'new_ex_occ1',
                            'new_ex_occ2',
                            'new_ex_occ3',
                            'new_ex_occ5_long',
                            'new_ex_occ6',
                            'new_ex_occ7.1',
                            'new_student_center1',
                            'new_student_center2',
                            'new_student_center3',
                            'new_student_center4',
                            'new_student_center_no_occ',
                            'new_ye_no_occ',
                            'new_ye_occ',
                            'one_book_move',
                            'rose1.2',
                            'static_sign1',
                            'studentcenter2.1',
                            'studentcenter3.1',
                            'studentcenter3.2',
                            'three_people',
                            'toy_car_no',
                            'toy_car_occ',
                            'toy_green_occ',
                            'toy_mo_occ',
                            'toy_no',
                            'toy_no_occ',
                            'toy_wg_no_occ',
                            'toy_wg_occ',
                            'toy_wg_occ1',
                            'toy_yellow_no',
                            'tracking4',
                            'tracking7.1',
                            'two_book',
                            'two_dog_occ1',
                            'two_people_1.1',
                            'two_people_1.2',
                            'two_people_1.3',
                            'walking_no_occ',
                            'walking_occ1',
                            'walking_occ_long',
                            'wdog_no1',
                            'wdog_occ3',
                            'wr_no',
                            'wr_no1',
                            'wr_occ2',
                            'wuguiTwo_no',
                            'zball_no1',
                            'zball_no2',
                            'zball_no3',
                            'zballpat_no1'
                            ]
        #print(self.base_path)
        #print('length of dataset sequence list: %d'%len(sequence_list))

        return sequence_list
