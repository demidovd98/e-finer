import os
import torch
import torch.cuda
import yaml
from easydict import EasyDict
import numpy as np
import random
from utils.fileios import mkdir_if_missing


def setup_config(config_file_env: str, config_file_expt: str):
    with open(config_file_env, 'r') as stream:
        config_env = yaml.safe_load(stream)

    with open(config_file_expt, 'r') as stream:
        config_expt = yaml.safe_load(stream)

    cfg_env = EasyDict()
    cfg_expt = EasyDict()

    # Copy
    for k, v in config_env.items():
        cfg_env[k] = v

    for k, v in config_expt.items():
        cfg_expt[k] = v

    # Init. configuration
    #   |- device
    cfg_expt['host'] = cfg_env['host']
    cfg_expt['num_workers'] = cfg_env['num_workers']

    if torch.cuda.is_available():
        cfg_expt['device'] = "cuda"
        cfg_expt['device_count'] = f"{torch.cuda.device_count()}"
        cfg_expt['device_id'] = f"{torch.cuda.current_device()}"
    else:
        cfg_expt['device'] = "cpu"

    #   |- file paths
    if cfg_expt['dataset_name'] == "bird":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "CUB_200_2011/CUB_200_2011/")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "bird200")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "dog":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "dogs_120")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "dog120")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "pet":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "pet_37")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "pet37")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "flower":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "flowers_102")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "flower102")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "car":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "car_196")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "car196")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "food":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "food_101")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "food101")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "place":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "place_365")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "place365")
        mkdir_if_missing(cfg_expt['expt_dir'])
    elif cfg_expt['dataset_name'] == "pokemon":
        cfg_expt['data_dir'] = os.path.join(cfg_env['data_root'], "pokemon")
        cfg_expt['expt_dir'] = os.path.join(cfg_env['expt_root'], "pokemon")
        mkdir_if_missing(cfg_expt['expt_dir'])
    else:
        raise NameError(f"{cfg_expt['dataset_name']} is a wrong dataset name")

    #for Stage Guess
    cfg_expt['expt_dir_guess'] = cfg_expt['expt_dir']
    mkdir_if_missing(cfg_expt['expt_dir_guess'])
    cfg_expt['path_llm_replies_raw'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                    f"{cfg_expt['dataset_name']}_llm_replies_raw")
    cfg_expt['path_llm_replies_jsoned'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                       f"{cfg_expt['dataset_name']}_llm_replies_jsoned")
    cfg_expt['path_llm_gussed_names'] = os.path.join(cfg_expt['expt_dir_guess'],
                                                     f"{cfg_expt['dataset_name']}_llm_gussed_names")

    #   |- data augmentation
    return cfg_expt


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True