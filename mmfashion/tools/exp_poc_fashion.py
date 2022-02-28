"""
    PoC Fashion (Category prediction)
    Training (Avalanche) loop
"""

"mmFashion import"

"Avalanche import"
import avalanche
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Cumulative, Replay, JointTraining

"Pytorch import"
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

"Python Files"
from deep_fashion_cate_attr import DeepFashion
from utils_config import DatasetSetting
from global_cate_fashion_predictor import GlobalCatePredictorFashion


"""
Function main
"""
def main():
    "Configs"
    # dataset
    data_cfg = DatasetSetting()
    # GPU | Device
    cuda = 0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")
    # general config CL experiment
    # TODO: use args or make a class in utils_config!
    # https://github.com/ContinualAI/reproducible-continual-learning/blob/main/strategies/iCARL/experiment.py

    "Fashion - Scenario and Benchmarck"
    # data loader (TR)
    dataset = DeepFashion(data_cfg.data['img_path'], data_cfg.data['img_cate_file'], data_cfg.data['img_bbox_file'], data_cfg.img_size)
    print('dataset loaded')
    # data split
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print('dataset splitted')
    # build Benchmarks - Use nc_benchmark to setup the scenario and benchmark
    # TODO

    "Fashion - build model"
    # TODO

    "Fashion - build the Evaluation plugin (Avalanche)"
    # TODO

    " Fashion - CREATE THE STRATEGY INSTANCE (Replay)"
    # TODO

    "Print (DEBUG)"
    # Dataset
    print('Dataset len= ', len(dataset))
    print("TR len= ", len(train_dataset))
    print("VAL len= ", len(val_dataset))
    print("target TR example: ", train_dataset[1][1])
    print("img TR example: ", train_dataset[1][0])
    print("target VAL example: ", val_dataset[1][1])

    "Fashion - TRAINING LOOP"
    # TODO



"""
RUN
"""
if __name__ == '__main__':
    main()
