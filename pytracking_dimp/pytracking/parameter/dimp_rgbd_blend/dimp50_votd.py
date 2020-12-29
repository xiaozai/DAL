from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 18*16#18*16
    params.image_sample_size_human = 32*16#18*16
    params.search_area_scale = 5

    # Learning parameters
    params.sample_memory_size = 250#50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25#0.25
    params.train_skipping = 10#20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10#10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1
    params.update_classifier_initial=20
    params.update_classifier_initial_iter=2

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    # params.augmentation = {'fliplr': True,
    #                        'rotate': [10, -10, 45, -45],
    #                        'blur': [(3,1), (1, 3), (2, 2)],
    #                        'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
    #                        'dropout': (2, 0.2)}
    params.augmentation = {'fliplr': True,
                       'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45,-45, -60, 60],
                       'blur': [(2, 0.2), (0.2, 2), (3,1), (1, 3), (2, 2)],
                       'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                       'dropout': (7, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25#0.35#0.25
    params.distractor_threshold = 0.8#0.8
    params.hard_negative_threshold = 0.5#0.5

    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = False

    #redetection parameters
    params.num_history=5
    params.frames_true_validd=100
    params.target_refound_threshold=params.target_not_found_threshold
    params.target_forcerefound_threshold=params.target_not_found_threshold+0.03 #recover from redtection model, even the valid_d is false
    params.threshold_updatedepth=0.80
    params.threshold_force_redetection=0.20
    params.threshold_allowupdateclassifer=0.30

    # IoUnet parameters
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1
    params.rotate_init_random_boxes=False

    params.net = NetWithBackbone(net_path='dimp50.pth',use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    #depth parameters
    params.use_depth_channel=True
    params.ptb_setting = False
    params.votd_setting=True
    params.stc_setting =False
    params.threshold_bhatta=0.8

    return params
