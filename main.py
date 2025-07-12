import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.3', config_path='./config', config_name='config')
def main(cfg: DictConfig):
    print(cfg)
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())

    from method_series.dpm_trainer import Trainer
    from method_series_large.dpm_trainer import Trainer_large

    # if cfg.data.data in ['ogbn-arxiv', 'ogbn-products']:
    if cfg.data.data in ['ogbn-products']:
        trainer = Trainer_large(cfg)
    else:
        trainer = Trainer(cfg)

    trainer.train(ts)

if __name__ == '__main__':
    main()
