"""
Test - benchmark
"""

# mmFashion import
from __future__ import division
import argparse

from mmcv import Config

from mmfashion.apis import (get_root_logger, init_dist, set_random_seed,
                            train_predictor)
from mmfashion.datasets import get_dataset
from mmfashion.models import build_predictor
from mmfashion.utils import init_weights_from

#Pytorch import
import torch
# Avalanche import
import avalanche
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils import AvalancheTensorDataset


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
Utilis function (DATA Manage)
Param: 
Return: 
"""
def get_xy(data_deepfashion):
    for i, example in enumerate(data_deepfashion): # example perche'? :)
        x_tmp = data_deepfashion[i]['img']
        y_tmp = data_deepfashion[i]['cate']
        if i == 0:
            x = x_tmp
            y = y_tmp
        else:
            x = torch.cat((x, x_tmp), 0)
            y = torch.cat((y, y_tmp), 0)
        print(i) #debug
    return x, y


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

    ## Print for test (Dataset)
    print("cfg.data.train: ", cfg.data.train)
    print("dataset type: ", type(dataset))
    print("dataset: ", dataset)
    print(dataset.__len__())
    print("dataset: ", dataset[1])
    print()

    ## data = {'img': img, 'attr': label, 'cate': cate, 'landmark': landmark}
    ## Prepare dataset for Avalanche benchmark
    x_data,y_data = get_xy(dataset) # TODO: here
    # Create the AvalancheDataset
    avalanche_dataset = AvalancheTensorDataset(x_data,y_data)
    print('avalanche_dataset: ', avalanche_dataset)
    print("x_data idx: ", x_data[1])
    print("y_data idx: ", y_data[1])
    # Given these two sets we can simply iterate them to get the examples one by one
    #for i, example in enumerate(dataset):
    #    pass
    #print("Num. examples processed: {}".format(i))

    # or use a Pytorch DataLoader
    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    #print("train_loader type: ", type(train_loader))
    #print("train_loader: ", train_loader)
    #for i, mini_ex in enumerate(train_loader):
    #    print("mini_ex: ", mini_ex)
        #pass
    #print("Num. mini-batch processed: {}".format(i))


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
    #TODO: use dataset as input to avalanche benchmark!
    #scenario = nc_benchmark(dataset, dataset, n_experiences=10, shuffle=True, seed=1234, task_labels=False)

    ## Print for test
    #TODO

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
