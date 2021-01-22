from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation

import numpy as np
import cv2 as cv

class DiMP(BaseTracker):

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            print('Song in pytracking.tracker.dimp_rgbd_blend.dimp.py line 19, initialize_features ..')
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image,depth, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not hasattr(self.params, 'device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net # ltr.models.tracking.dimpnet

        # Convert image
        im = numpy_to_torch(image)
        im_d=numpy_to_torch(depth)

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])


        # #initial min and max depth in ref bbox
        # im_d_bbox=im_d[:,:, np.int16(state[1]):np.int16(state[1]+state[3]+10), np.int16(state[0]):np.int16(state[0]+state[2]+10) ]
        # #print(im_d_bbox[0,0,20,:])
        # self.max_depth = torch.max(im_d_bbox)
        # im_d_bbox[im_d_bbox==0]=self.max_depth
        # self.min_depth = torch.min(im_d_bbox)
        # #self.mean_depth= torch.mean(im_d_bbox)
        # self.median_depth= (self.max_depth+self.min_depth)/2.0
        # self.mask_seg  = im_d_bbox.le(self.median_depth)
        # depth_object   = torch.masked_select(im_d_bbox, self.mask_seg)
        # self.depth_mean= torch.mean(depth_object)
        # self.depth_variance=(torch.max(depth_object)-torch.min(depth_object))/2.0
        #print('depth', depth.shape)

        init_bbox = state.copy()
        init_bbox[0] = init_bbox[0] - 0.5 * init_bbox[2]
        init_bbox[1] = init_bbox[1] - 0.5 * init_bbox[3]
        depth_init = self.avg_depth(depth[:,:,0], init_bbox)
        depth_hist_init=self.hist_depth(depth[:,:,0], init_bbox)

        self.history_info ={'depth':[depth_init],\
        'depth_hist':[depth_hist_init], \
        'velocity':[0.0], \
        'ratio_bbox':[self.target_sz[0]/self.target_sz[1]],\
        'target_sz':[self.target_sz],\
        'pos':[self.pos] \
        }

        print('init depth', depth_init, 'bhatta_depth', self.bhatta(depth_hist_init,depth_hist_init),'ratio_bbox',self.history_info['ratio_bbox'][-1] )

        ratio_sz=depth_init/1.0
        ratio_sz=max(1.0, ratio_sz)
        ratio_sz=min(2.0, ratio_sz)
        #Set sizes only for princeton benchmark
        if hasattr(self.params, 'ptb_setting'):
            if self.params.ptb_setting:
                if (state[2]/state[3] <= 2.5 and state[3]/state[2] <= 2.5): #or state[2]*state[3]<=9000 :# not a human, or, it is so tiny
                    ratio_sz=1
                else:
                    pass
        if hasattr(self.params, 'votd_setting'):
            if self.params.votd_setting:
                ratio_sz=1
        if hasattr(self.params, 'stc_setting'):
            if self.params.stc_setting:
                ratio_sz=1

        #ratio_sz=1
        if self.params.debug>=1:
            print(['initial size:', state[2], state[3]])
            print(['ratio_sz', ratio_sz])
        sz = int(math.floor(ratio_sz*18))*16





        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area
        #print(self.target_sz, self.params.search_area_scale)
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt() #1.8176
        self.first_target_scale=self.target_scale
        self.target_scale_redetection=self.target_scale
        self.redetection_mode=False

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        if self.params.debug>=1:
            print(['self.base_target_sz',self.base_target_sz,'self.target_scale', self.target_scale])
        # Setup scale factors
        if not hasattr(self.params, 'scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)



        # Extract and transform sample
        init_backbone_feat, patches, patches_d = self.generate_init_samples(im,im_d)



        # Initialize classifier
        self.init_classifier(init_backbone_feat, patches_d)

        # Initialize IoUNet
        if getattr(self.params, 'use_iou_net', True):
            self.init_iou_net(init_backbone_feat, patches_d)

        out = {'time': time.time() - tic}
        return out


    def track(self, image, depth) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)
        im_d=numpy_to_torch(depth)

        # print('frame_num : ', self.frame_num)
        # print('im shape : ', im.shape)
        # print('im_d shape : ', im_d.shape)
        self.flag='noaction'
        self.valid_d=True
        self.dimpfail_thistime=False

        # ------- DIMP LOCALIZATION ------- #
        if self.redetection_mode==False:
            # print(['pos',self.pos])
            # print(['get_centered_sample_pos',self.get_centered_sample_pos()])
            # Extract backbone features
            backbone_feat, sample_coords = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                          self.target_scale * self.params.scale_factors,#1.82
                                                                          self.img_sample_sz)#18*16

            # Extract classification features
            test_x = self.get_classification_features(backbone_feat)

            # Location of sample
            sample_pos, sample_scales = self.get_sample_location(sample_coords)
            #print(sample_scales)

            patches_d = self.extract_patches(im_d, self.get_centered_sample_pos(),self.target_scale * self.params.scale_factors,self.img_sample_sz)#18*16
            patches_d = patches_d[:,0,:,:]
            patches_d = patches_d.view(patches_d.shape[0],1,patches_d.shape[1], patches_d.shape[2])
            # Compute classification scores
            scores_raw = self.classify_target(test_x, patches_d)
            self.score_map  =scores_raw
            self.score_map_stationary= self.classify_target(test_x, None)

            # Localize the target
            translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_scales)
            new_pos = sample_pos[scale_ind,:] + translation_vec
            self.translation_vec=translation_vec
            #print('translation_vec', translation_vec)


            self.debug_info['flag'] = flag
            self.flag=flag
            # Update position and scale
            if flag != 'not_found':
                #we found the target, we reset the s
                self.target_scale_redetection=self.target_scale
                if getattr(self.params, 'use_iou_net', True):
                    update_scale_flag = getattr(self.params, 'update_scale_when_uncertain', True) or flag != 'uncertain'
                    if getattr(self.params, 'use_classifier', True):
                        self.update_state(new_pos)
                    self.net.bb_regressor.test_depths=patches_d
                    self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
                elif getattr(self.params, 'use_classifier', True):
                    self.update_state(new_pos, sample_scales[scale_ind])



            # if flag=='not_found':
            #     self.redetection_mode=True

        # ------- end of DIMP LOCALIZATION ------- #

        #measure the average depth
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))
        dimp_bbox = new_state.tolist()
        new_d     = self.avg_depth(depth[:,:,0], dimp_bbox)
        self.valid_d   = self.valid_depth(new_d, self.history_info['depth'][-1])

        new_dhist = self.hist_depth(depth[:,:,0], dimp_bbox)
        new_bhatta_depth=self.bhatta(new_dhist, self.history_info['depth_hist'][-1])
        # self.valid_d   = new_bhatta_depth>=self.params.threshold_bhatta
        self.valid_d = np.all([self.bhatta(new_dhist,dhist)>=self.params.threshold_bhatta for dhist in self.history_info['depth_hist']])

        new_ratio = self.target_sz[0]/self.target_sz[1]
        mean_historyratios=np.mean(self.history_info['ratio_bbox'])
        mean_h =np.mean([bbox[0] for bbox in self.history_info['target_sz']])
        mean_w =np.mean([bbox[1] for bbox in self.history_info['target_sz']])
        mean_target_sz=[mean_h, mean_w]
        change_ratio = abs(new_ratio-mean_historyratios)/mean_historyratios


        # if len(self.history_info['pos'])>=2:
        #     move_y_0=self.history_info['pos'][-1][0]-self.history_info['pos'][-2][0]
        #     move_x_0=self.history_info['pos'][-1][1]-self.history_info['pos'][-2][1]
        #     move_y  =self.pos[0]-self.history_info['pos'][-1][0]
        #     move_x  =self.pos[1]-self.history_info['pos'][-1][1]
        #     change_distance=np.sqrt((move_x-move_x_0)**2+(move_y-move_y_0)**2)
        #     if change_distance>=100:
        #         #print('abrupt change_distance', change_distance)
        #         mean_y =np.mean([yx[0] for yx in self.history_info['pos']])
        #         mean_x =np.mean([yx[1] for yx in self.history_info['pos']])
        #         mean_yx=[mean_y, mean_x]
        #         self.pos=torch.FloatTensor(mean_yx)



        #print('change_ratio', change_ratio, new_ratio, mean_historyratios)
        if change_ratio>0.50:#some bug?
            #print('abrupt bbox ratio change, use the past target_sz',self.history_info['target_sz'])
            self.target_sz=torch.FloatTensor(mean_target_sz)#self.history_info['target_sz'][-1]



        if self.frame_num<self.params.frames_true_validd:
            self.valid_d=True

        if self.params.debug>=1:
            print(self.frame_num, 'score',self.score_map.max(),'bhatta_depth', new_bhatta_depth,'average depth', new_d, 'history_depth', self.history_info['depth'][-1], 'valid_d', self.valid_d)
            print('target_sz', self.target_sz, 'ratio_bbox', self.target_sz[0]/self.target_sz[1], 'base_target_sz', self.base_target_sz, 'target_scale', self.target_scale)
            # if new_ratio>2:
            #     input()
        #if confident enough, we update the average depths and the velocity
        if (self.score_map.max()>=self.params.threshold_updatedepth) or (self.score_map.max()>=self.params.target_not_found_threshold and self.valid_d):
            self.history_info['depth'].append(new_d)
            if len(self.history_info['depth'])>self.params.num_history:#5:
                self.history_info['depth']=self.history_info['depth'][-self.params.num_history:]

            self.history_info['depth_hist'].append(new_dhist)
            if len(self.history_info['depth_hist'])>self.params.num_history:#5:
                self.history_info['depth_hist']=self.history_info['depth_hist'][-self.params.num_history:]

            self.history_info['target_sz'].append(self.target_sz)
            if len(self.history_info['target_sz'])>self.params.num_history:#3:
                self.history_info['target_sz']=self.history_info['target_sz'][-self.params.num_history:]

            new_ratio=self.target_sz[0]/self.target_sz[1]
            self.history_info['ratio_bbox'].append(new_ratio)
            if len(self.history_info['ratio_bbox'])>self.params.num_history:#3:
                self.history_info['ratio_bbox']=self.history_info['ratio_bbox'][-self.params.num_history:]

            self.history_info['pos'].append(self.pos)
            if len(self.history_info['pos'])>self.params.num_history:#3:
                self.history_info['pos']=self.history_info['pos'][-self.params.num_history:]


            # #speed
            # self.history_info['velocity'].append(math.sqrt(self.translation_vec[0]*self.translation_vec[1]))
            # if len(self.history_info['velocity'])>5:
            #     self.history_info['velocity']=self.history_info['velocity'][-5:]
            # print('update speed', self.history_info['velocity'][-1])


        #if not confident, and depth is not valid, we switch to redetection mode
        if (self.score_map.max()<self.params.threshold_force_redetection) or (self.flag=='not_found' and self.valid_d==False):
            self.redetection_mode=True
            self.dimpfail_thistime=True
            if self.params.debug>=1:
                #input()
                print('.........attention! next iter entering redetection model', self.frame_num, self.score_map.max())

        # ------- redetection module ------- #
        if  self.redetection_mode:
            self.target_scale_redetection=self.target_scale_redetection*1.05 #slowing enlarge this area to the object
            self.target_scale_redetection=max(self.target_scale_redetection, self.min_scale_factor)
            self.target_scale_redetection=min(self.target_scale_redetection, 2*self.first_target_scale)
            #get backbone feat of im in a larger region

            #img_sample_sz is made bigger correspondingly
            # sz = int(math.floor(self.target_scale_redetection/self.target_scale*18))*16
            # img_sample_sz_rede = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
            # backbone_feat_re, sample_coords_re = self.extract_backbone_features(im, self.get_centered_sample_pos(),self.target_scale_redetection*self.params.scale_factors,img_sample_sz_rede)

            backbone_feat_re, sample_coords_re = self.extract_backbone_features(im, self.get_centered_sample_pos(),self.target_scale_redetection*self.params.scale_factors,self.img_sample_sz)

            test_x_re = self.get_classification_features(backbone_feat_re)
            sample_pos_re, sample_scales_re = self.get_sample_location(sample_coords_re)
            scores_re = self.classify_target(test_x_re, None)
            self.score_map = scores_re
            translation_vec_re, scale_ind_re, s_re, flag_re = self.localize_target(scores_re, sample_scales_re)
            new_pos_re = sample_pos_re[scale_ind_re,:] + translation_vec_re
            self.update_state(new_pos_re)

            self.redetection_mode=True
            if scores_re.max()>=self.params.target_refound_threshold and self.valid_d:
                self.redetection_mode=False
            if scores_re.max()>=self.params.target_forcerefound_threshold:
                self.redetection_mode=False

            if self.redetection_mode==False: #target refound
                if self.params.debug>=1:
                    print('.........attention!, target refound, next iter moving to dimp tracking',scores_re.max(), self.target_scale_redetection)

                #update_scale_flag_re = getattr(self.params, 'update_scale_when_uncertain', True) or flag_re != 'uncertain'
                #self.refine_target_box(backbone_feat_re, sample_pos_re[scale_ind_re,:], sample_scales_re[scale_ind_re], scale_ind_re, update_scale_flag_re)

                #second stage redetection
                backbone_feat_re2, sample_coords_re2 = self.extract_backbone_features(im, self.get_centered_sample_pos(),self.target_scale*self.params.scale_factors,self.img_sample_sz)
                test_x_re2 = self.get_classification_features(backbone_feat_re2)
                #print('redetection 2: ', self.frame_num, 'img_sample_sz', self.img_sample_sz, 'test_x_re', test_x_re2.shape)
                sample_pos_re2, sample_scales_re2 = self.get_sample_location(sample_coords_re2)
                scores_re2 = self.classify_target(test_x_re2, None)
                self.score_map = scores_re2
                translation_vec_re2, scale_ind_re2, s_re2, flag_re2 = self.localize_target(scores_re2, sample_scales_re2)
                new_pos_re2 = sample_pos_re2[scale_ind_re2,:] + translation_vec_re2
                self.update_state(new_pos_re2)
                #update_scale_flag_re2 = getattr(self.params, 'update_scale_when_uncertain', True) or flag_re2 != 'uncertain'
                #self.refine_target_box(backbone_feat_re2, sample_pos_re2[scale_ind_re2,:], sample_scales_re2[scale_ind_re2], scale_ind_re2, update_scale_flag_re2)

                #print('redetectin 2: reset target_scale_redes', self.target_scale_redetection, self.target_scale)
                self.target_scale_redetection=self.first_target_scale
                self.target_scale=self.first_target_scale
                self.redetection_mode=False

        # ------- end for redetection module ------- #



        # ------- UPDATE ------- #

        update_flag = self.flag not in ['not_found', 'uncertain', 'noaction']
        hard_negative = (self.flag == 'hard_negative')
        learning_rate = getattr(self.params, 'hard_negative_learning_rate', None) if hard_negative else None
        goodscore_flag = self.score_map.max()>=self.params.threshold_allowupdateclassifer

        if (getattr(self.params, 'update_classifier', False) and self.frame_num<=self.params.update_classifier_initial) or \
         (getattr(self.params, 'update_classifier', False) and update_flag and goodscore_flag and self.redetection_mode==False and self.valid_d):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Get the train depth
            train_depths = patches_d
            if train_x.shape[2]!=train_depths.shape[2] or train_x.shape[3]!=train_depths.shape[3]:
                train_depths=F.upsample(train_depths, size=(train_x.shape[2], train_x.shape[3]), mode='bilinear')

            # Update the classifier model
            self.update_classifier(train_x, target_box, train_depths, learning_rate, s[scale_ind,...])

        # Set the pos of the tracker to iounet pos
        if getattr(self.params, 'use_iou_net', True) and self.flag != 'not_found' and hasattr(self, 'pos_iounet') and self.redetection_mode==False and self.valid_d:
            self.pos = self.pos_iounet.clone()

        max_score = torch.max(self.score_map).item()
        self.debug_info['max_score'] = max_score

        if self.visdom is not None:
            self.visdom.register(self.score_map, 'heatmap', 2, 'Score Map')
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 3:
            if hasattr(self,'ind_frame'):
                show_tensor(self.score_map, 5, title='confidence_%d.png'%self.ind_frame)
                if hasattr(self, 'score_map_stationary'):
                    ratio=self.score_map.max()/self.score_map_stationary.max()
                    self.score_map_stationary = self.score_map_stationary*ratio
                    show_tensor(self.score_map_stationary, 6, title='confidence_%d_stationary.png'%self.ind_frame)

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.ptb_setting:
            if self.flag == 'not_found': #e.g. occluted, out of view
                out = {'target_bbox': [0, 0, 0, 0]}
            else:
                out = {'target_bbox': new_state.tolist()}
        else:
            out = {'target_bbox': new_state.tolist()}
        return out

    def hist_depth(self, depth_im, bbox):
        #bbox [x0,y0,w,h]
        # get center around depth
        cx, cy = bbox[0]+bbox[2]/2, bbox[1] + bbox[2]/2

        def shrink_box(bbox, mh, mw, r=0.4):
            w, h = (1-r) * bbox[2], (1-r) * bbox[3]
            if w<2 or h<2:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                return [lx, ly, bbox[2], bbox[3]]
            else:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                w = min(w, mw - w)
                h = min(h, mh - h)
                return [lx, ly, w, h]

        h, w = depth_im.shape
        shrink_box = shrink_box(bbox, h, w, r=0.4)
        shrink_box = np.asarray(shrink_box).astype(np.int16)
        depth_im[np.isnan(depth_im)]=8.0
        depth_inbox = depth_im[shrink_box[1]:(shrink_box[1]+shrink_box[3]),shrink_box[0]:(shrink_box[0]+shrink_box[2])]
        depth_inbox = depth_inbox.flatten()
        hist,_=np.histogram(depth_inbox, bins=np.arange(0,8,0.1), density=True)

        return hist

    def bhatta(self, hist1, hist2):
        hist1=hist1/np.sum(hist1)
        hist2=hist2/np.sum(hist2)
        return sum(np.sqrt(np.multiply(hist1,hist2)))

    def avg_depth(self, depth_im, bbox):
        #bbox [x0,y0,w,h]
        # get center around depth
        cx, cy = bbox[0]+bbox[2]/2, bbox[1] + bbox[2]/2

        def shrink_box(bbox, mh, mw, r=0.4):
            w, h = (1-r) * bbox[2], (1-r) * bbox[3]
            if w<2 or h<2:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                return [lx, ly, bbox[2], bbox[3]]
            else:
                lx = max(cx-w/2, 0)
                ly = max(cy-h/2, 0)
                w = min(w, mw - w)
                h = min(h, mh - h)
                return [lx, ly, w, h]

        h, w = depth_im.shape
        #print ("avg_depth h: {} w: {}".format(h, w))
        shrink_box = shrink_box(bbox, h, w, r=0.4)
        shrink_box = np.asarray(shrink_box).astype(np.int16)
        avg_depth = np.mean(depth_im[shrink_box[1]:(shrink_box[1]+shrink_box[3]), \
                shrink_box[0]:(shrink_box[0]+shrink_box[2])])

        if np.isnan(avg_depth):
            return -1.
        return avg_depth

    def valid_depth(self, depth_a, history_depth):
        depth_margin = 0.6#600
        if depth_a>2:#2000:
            depth_percent = depth_margin / depth_a
        else:
            depth_percent = 0.4
        d = np.abs((depth_a-history_depth))
        if d == 0 or depth_a == 0:
            return False
        p = d/depth_a
        # if d>depth_margin or p > depth_percent:
        if p > depth_percent:
            return False
        else:
            return True

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList, depths):
        """Classify target by applying the DiMP filter."""
        # if hasattr(self.net.classifier,'target_mask'):
        #     print('target_mask',self.net.classifier.target_mask.shape)
        #     show_tensor(self.net.classifier.target_mask[0,0,:,:], 6, title='targetmask_%d.png'%self.ind_frame)

        depthcnn_online=self.net.settings.depthaware_for_classiferonline
        alpha=self.net.settings.depthaware_alpha
        if depths is not None:
            with torch.no_grad():
                scores = self.net.classifier.depthaware_classify(self.target_filter, sample_x, depths, alpha)
        else:
            with torch.no_grad():
                scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def localize_target(self, scores, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        if getattr(self.params, 'advanced_localization', False):
            return self.localize_advanced(scores, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        translation_vec = target_disp * (self.img_support_sz / self.feature_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / self.feature_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (self.feature_sz / self.img_support_sz)
        #print([sample_scale])
        #1.86
        #print([self.params.target_neighborhood_scale, (self.target_sz / sample_scale), (self.feature_sz / self.img_support_sz)])
        #[2.2, tensor([53.4149, 62.2256]), tensor([0.0625, 0.0625])=18/(18*16)]


        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0
        #print([tneigh_top, tneigh_bottom, tneigh_left, tneigh_right])
        #[5, 14, 5, 14]


        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / self.feature_sz) * sample_scale

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
            disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2
            #print([disp_norm1, disp_norm2, disp_threshold])

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz, getattr(self.params, 'border_mode', 'replicate'))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords

    def extract_patches(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz, getattr(self.params, 'border_mode', 'replicate'))
        return im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes, depths):
        with torch.no_grad():
            self.net.bb_regressor.train_depths=depths
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor, im_d:  torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        if getattr(self.params, 'border_mode', 'replicate') == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz).max().clamp(1)
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if getattr(self.params, 'use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)
        im_patches_d = sample_patch_transformed(im_d, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat,im_patches,im_patches_d

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList, train_d: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])
        self.training_samples_depth = TensorList(
            [d.new_zeros(self.params.sample_memory_size, d.shape[1], d.shape[2], d.shape[3]) for d in train_d])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x
        for ts, d in zip(self.training_samples_depth, train_d):
            ts[:d.shape[0],...] = d


    def update_memory(self, sample_x: TensorList, sample_d: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for train_samp_d, d, ind in zip(self.training_samples_depth, sample_d, replace_ind):
            train_samp_d[ind:ind+1,...] = d

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = getattr(self.params, 'init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = getattr(self.params, 'target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_iou_net(self, backbone_feat, depths):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes,depths)
        self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])



    def init_classifier(self, init_backbone_feat, depths):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)
        # cat x and depth together for augmentation
        channels_x = x.shape[1]
        depths=depths.to(x.device)
        depths = F.upsample(depths, size=(x.shape[2],x.shape[3]), mode='bilinear')
        x = torch.cat([x,depths],dim=1)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and getattr(self.params, 'use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        #18,18
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))

        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2
        #print(['output_sz', self.output_sz])

        # Construct output window
        self.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), self.output_sz.long()*self.params.effective_search_area / self.params.search_area_scale, centered=False).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug >= 3
        num_iter = getattr(self.params, 'net_opt_iter', None)

        # split x into x and depths
        depths = x[:,channels_x::,:,:]
        x      = x[:,0:channels_x,:,:].contiguous()
        depths = depths[:,0,:,:]
        depths = depths.view(depths.shape[0],1, depths.shape[1], depths.shape[2])
        #print(x.shape, depths.shape, target_boxes.shape)
        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            print('Song in pytracking.tracker.dimp_rgbd_blend.dimp.py line 899, initialize_classifier ...')
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, depths, None, num_iter=num_iter,
                                                                           compute_losses=plot_loss)
            import os
            import scipy
            if os.path.exists('./tracking_results/imgs'):
                target_filter=self.target_filter.cpu().numpy()
                scipy.io.savemat('./tracking_results/imgs/filter.mat',{'filter':target_filter})

        # Init memory
        if getattr(self.params, 'update_classifier', True):
            self.init_memory(TensorList([x]), TensorList([depths]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.stack(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss')
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training loss')


    def update_classifier(self, train_x, target_box, train_depths, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        self.update_memory(TensorList([train_x]), TensorList([train_depths]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = getattr(self.params, 'low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = getattr(self.params, 'net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = getattr(self.params, 'net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = getattr(self.params, 'net_opt_update_iter', None)

        if self.frame_num<=self.params.update_classifier_initial:
            num_iter = self.params.update_classifier_initial_iter



        plot_loss = self.params.debug >= 3

        if num_iter > 0:
            if self.params.debug>=1:
                print('update_classifier', num_iter)
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]
            samples_depth = self.training_samples_depth[0][:self.num_stored_samples[0],...]
            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter, samples, target_boxes, samples_depth, None,
                                                                                     sample_weight=sample_weights,
                                                                                     num_iter=num_iter,
                                                                                     compute_losses=plot_loss)


            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.stack(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss')
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training loss')

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        #rotate by 90 degree to generate more init boxes
        if self.params.rotate_init_random_boxes:
            init_boxes2 = init_boxes.clone()
            init_boxes2 = init_boxes2[:,[0,1,3,2]]
            init_boxes  = torch.cat([init_boxes, init_boxes2])



        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = getattr(self.params, 'iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)
        if self.params.visualization:
            print(['predicted_iou',predicted_iou])

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if getattr(self.params, 'use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def optimize_boxes(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()
