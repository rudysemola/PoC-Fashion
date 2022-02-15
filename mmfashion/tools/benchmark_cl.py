"""
Test - benchmark
"""

# mmFashion import
from __future__ import division
import argparse

from mmcv import Config

from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_predictor)

from mmfashion.models import build_predictor
from mmfashion.utils import init_weights_from

from deep_fashion_cate_attr import DeepFashion

#Pytorch import
import torch
# Avalanche import
import avalanche
from avalanche.benchmarks.generators import nc_benchmark


"""
Parse Arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a Fashion Attribute Predictor')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/attribute_predict_coarse/roi_predictor_vgg_attr.py')
    parser.add_argument('--work_dir', help='the dir to save logs and checkpoint')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training',
        default=True)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'mpi', 'slurm'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


"""
Utils
"""
def get_dataset(data_cfg):
    dataset = DeepFashion(data_cfg.img_path, data_cfg.img_file,
                          data_cfg.label_file, data_cfg.cate_file,
                          data_cfg.bbox_file, data_cfg.landmark_file,
                          data_cfg.img_size)

    return dataset


"""
Function main
"""
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # build model
    model = build_predictor(cfg.model)
    print('model built')

    if cfg.init_weights_from:
        model = init_weights_from(cfg.init_weights_from, model)

    # data loader
    dataset = get_dataset(cfg.data.train)
    print('dataset loaded')

    # train
    """
    train_predictor(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)
    """

    # build Benchmarks

    ## Use nc_benchmark to setup the scenario and benchmark
    scenario = nc_benchmark(dataset, dataset, n_experiences=10, shuffle=True, seed=1234, task_labels=False)

    ## Print for test
    ## Print for test (Dataset) DEBUG
    print("cfg.data.train: ", cfg.data.train)
    print("dataset type: ", type(dataset))
    print("dataset: ", dataset)
    print(dataset.__len__())
    print("dataset: ", dataset[1])
    print("dataset X: ", dataset[10][0])
    print("dataset Y: ", dataset[10][1])
    print(dataset.label_type)
    print()
    for i, example in enumerate(dataset):
        print(example)
        break
    print("Num. examples processed: {}".format(i))
    print()
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    for i, (x, y) in enumerate(train_loader):
        print(y)
        break
    print("Num. mini-batch processed: {}".format(i))
    ## data = {'img': img, 'attr': label, 'cate': cate, 'landmark': landmark}

    #print("scenario: ", scenario)
    """
    train_stream = scenario.train_stream
    print("train_stream: ", train_stream)
    for experience in train_stream:
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')
    """


"""
RUN
"""
if __name__ == '__main__':
    main()
