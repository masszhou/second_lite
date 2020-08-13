from comet_ml import Experiment
import fire
from tqdm import tqdm
import time
import torch

from utils.configure_tools import parse_cfg


def training(train_cfg_path: str,
             net_cfg_path: str,
             log_comet_ml=False):
    print('Train simplified SECOND')

    # ------------------------------------------------------------
    # 1. get parameters
    train_cfg = parse_cfg(train_cfg_path)
    net_cfg = parse_cfg(net_cfg_path)

    # ------------------------------------------------------------
    # 2. build dataset -> output sample dict

    # ------------------------------------------------------------
    # 3. build dataloader -> output batch tensor

    # ------------------------------------------------------------
    # 4. define network

    # ------------------------------------------------------------
    # 5. associate network model to a training instance
    # ToDo: resume training

    # ------------------------------------------------------------
    # 6. define logging enviroment

    # ------------------------------------------------------------
    # 7. start training phase


if __name__ == '__main__':
    fire.Fire(training)
