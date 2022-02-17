"""
    PoC Fashion (Category prediction)
    Training (Avalanche) loop
"""

"Avalanche import"
import avalanche
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Cumulative, Replay

"Pytorch import"
import torch
from torch.optim import SGD

# from torch.nn import CrossEntropyLoss #TODO: define LOSS

"Python Files"
from deep_fashion_cate_attr import DeepFashion
from utils_config import DatasetSetting


"""
Function main
"""
def main():
    "Config - Dataset"
    data_cfg = DatasetSetting()

    "Fashion - Scenario and Benchmarck"
    # data loader (TR)
    train_dataset = DeepFashion(data_cfg.data.train['img_path'], data_cfg.data.train['img_file'],
                          data_cfg.data.train['label_file'], data_cfg.data.train['cate_file'],
                          data_cfg.data.train['bbox_file'], data_cfg.data.train['landmark_file'],
                          data_cfg.data.train['img_size'])
    # dataset loader (Val)
    val_dataset = DeepFashion(data_cfg.data.val['img_path'], data_cfg.data.val['img_file'],
                          data_cfg.data.val['label_file'], data_cfg.data.val['cate_file'],
                          data_cfg.data.val['bbox_file'], data_cfg.data.val['landmark_file'],
                          data_cfg.data.val['img_size'])
    print('dataset loaded')

    # build Benchmarks
    ## Use nc_benchmark to setup the scenario and benchmark
    scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=5, shuffle=True, seed=1234, task_labels=False)

    # Print for dataset and benchmark test (DEBUG)
    print("Tot len train DT: ", len(train_dataset))
    print("Tot len val DT: ", len(val_dataset))
    print()
    print("scenario: ", scenario)
    train_stream = scenario.train_stream
    val_stream  = scenario.test_stream
    print("train_stream: ", train_stream)
    for experience in train_stream:
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')
        print("No. Classes: ", len(experience.classes_in_this_experience))
        print("Current Classes: ", experience.classes_in_this_experience)
        print()
    for experience_val in val_stream:
        t = experience_val.task_label
        exp_id = experience_val.current_experience
        validation_dataset = experience_val.dataset
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(validation_dataset), 'patterns')
        print("No. Classes: ", len(experience_val.classes_in_this_experience))
        print("Current Classes: ", experience_val.classes_in_this_experience)
        print()



"""
RUN
"""
if __name__ == '__main__':
    main()
