from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.mobiface_path = ''
    settings.network_path = '/home/qian2/qian2_old/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.ptb_path = '/home/qian2/qian2_old/Princeton_RGBD/ValidationSet'#ValidationSet'
    settings.results_path = '/home/qian2/qian2_old/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''

    settings.vot_path = '/home/qian2/qian2_old/VOT2018'
    settings.laval_6d_path='/home/qian2/qian2_old/laval_6d/fulldata/'

    return settings
