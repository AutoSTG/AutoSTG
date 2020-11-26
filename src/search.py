import logging

logging.basicConfig(level=logging.INFO)

import random
import numpy as np
import torch
from setting import config as cfg
from data.dataset import TrafficDataset
from model.nas import AutoSTG
from run_manager import RunManager


def system_init():
    """ Initialize random seed. """
    random.seed(cfg.sys.seed)
    np.random.seed(cfg.sys.seed)
    torch.manual_seed(cfg.sys.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def main(num_epoch):
    system_init()

    # load data
    dataset = TrafficDataset(
        path=cfg.data.path,
        train_prop=cfg.data.train_prop,
        valid_prop=cfg.data.valid_prop,
        num_sensors=cfg.data.num_sensors,
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        batch_size_per_gpu=cfg.data.batch_size_per_gpu,
        num_gpus=1  # torch.cuda.device_count()
    )

    net = AutoSTG(
        in_length=cfg.data.in_length,
        out_length=cfg.data.out_length,
        node_hiddens=[dataset.node_fts.shape[1], ] + cfg.model.node_hiddens,
        edge_hiddens=[dataset.adj_mats.shape[2], ] + cfg.model.edge_hiddens,
        in_channels=cfg.data.in_channels,
        out_channels=cfg.data.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        skip_channels=cfg.model.skip_channels,
        end_channels=cfg.model.end_channels,
        layer_names=cfg.model.layer_names,
        num_mixed_ops=cfg.model.num_mixed_ops,
        candidate_op_profiles=cfg.model.candidate_op_profiles
    )

    print('# of weight parameters', len(list(net.weight_parameters())))
    print('# of arch parameters', len(list(net.arch_parameters())))
    print('# of parameters', len(list(net.parameters())))

    run_manager = RunManager(
        name=cfg.model.name,
        net=net,
        dataset=dataset,

        arch_lr=cfg.trainer.arch_lr,
        arch_lr_decay_milestones=cfg.trainer.arch_lr_decay_milestones,
        arch_lr_decay_ratio=cfg.trainer.arch_lr_decay_ratio,
        arch_decay=cfg.trainer.arch_decay,
        arch_clip_gradient=cfg.trainer.arch_clip_gradient,

        weight_lr=cfg.trainer.weight_lr,
        weight_lr_decay_milestones=cfg.trainer.weight_lr_decay_milestones,
        weight_lr_decay_ratio=cfg.trainer.weight_lr_decay_ratio,
        weight_decay=cfg.trainer.weight_decay,
        weight_clip_gradient=cfg.trainer.weight_clip_gradient,

        num_search_iterations=cfg.trainer.num_search_iterations,
        num_search_arch_samples=cfg.trainer.num_search_arch_samples,
        num_train_iterations=cfg.trainer.num_train_iterations,

        criterion=cfg.trainer.criterion,
        metric_names=cfg.trainer.metric_names,
        metric_indexes=cfg.trainer.metric_indexes,
        print_frequency=cfg.trainer.print_frequency,

        device_ids=[0],  # range(torch.cuda.device_count())
    )

    run_manager.load(mode='search')
    run_manager.search(num_epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    cfg.load_config(args.config)
    main(args.epoch)
