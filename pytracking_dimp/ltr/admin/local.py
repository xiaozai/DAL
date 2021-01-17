class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yan/Data2/2D-Tracking/DAL/training/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ptb_dir = '/home/yan/Data1/RGBD_benchmarks/Princeton_RGBD/ValidationSet/ValidationSet/'
