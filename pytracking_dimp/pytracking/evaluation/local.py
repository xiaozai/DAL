from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/yan/Data2/DeT-models/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = '/home/yan/Data2/DAL/results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.votd_path = '/home/yans/Datasets/CDTB'
    settings.depthtrack_path = '/home/yan/Data2/ICCV2021-RGBD-DepthTrack/sequences/'

    return settings
