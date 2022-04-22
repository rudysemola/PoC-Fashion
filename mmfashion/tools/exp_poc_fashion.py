"""
    PoC Fashion (Category prediction)
    Training (Avalanche) loop
"""

"script import"
import argparse

"Avalanche import"

import avalanche
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Cumulative, Replay, JointTraining


"Pytorch import"
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

"Python Files"
from deep_fashion_cate_attr import DeepFashion
from utils_config import DatasetSetting, Timer
from global_cate_fashion_predictor import GlobalCatePredictorFashion
from topk_acc import *


"""
parse_args function
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description='PoC - Train a Fashion Category Predictor in DeepFashion')
    parser.add_argument('--strategy', help='strategy alg')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--cuda', help='in Multi-GPU choose che GPU', default=0)
    parser.add_argument('--memory_size', help='CL Replay hyper-param', default=10000)
    args = parser.parse_args()
    return args


"""
Function main
"""
def main():
    "Configs & Parser"
    # parser
    args = parse_args()
    # TIME
    timer = Timer()
    # dataset
    data_cfg = DatasetSetting()
    # GPU | Device
    ## (if multi-GPU -> 0 is Replay | 1 is Cumulative)
    cuda = int(args.cuda)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")
    # epochs
    epochs = int(args.epochs)
    # memory_size
    memory_size = int(args.memory_size)


    "Fashion - Scenario and Benchmarck"
    # data loader (TR)
    dataset = DeepFashion(data_cfg.data['img_path'], data_cfg.data['img_cate_file'], data_cfg.data['img_bbox_file'], data_cfg.img_size)
    print('dataset loaded')
    # data split
    dataset_size = len(dataset)
    val_size = int(0.3 * dataset_size)
    train_size = dataset_size - val_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print('dataset splitted')
    # build Benchmarks over 3 run - Use nc_benchmark to setup the scenario and benchmark
    scenario1 = nc_benchmark(train_dataset, val_dataset, n_experiences=10, shuffle=True, seed=1, task_labels=False, per_exp_classes={0:10})
    scenario2 = nc_benchmark(train_dataset, val_dataset, n_experiences=10, shuffle=True, seed=50, task_labels=False, per_exp_classes={0: 10})
    scenario3 = nc_benchmark(train_dataset, val_dataset, n_experiences=10, shuffle=True, seed=100, task_labels=False, per_exp_classes={0: 10})
    scenario_list = [scenario1, scenario2, scenario3]

    "Fashion - build model"
    model = GlobalCatePredictorFashion(num_classes=50, pretrained='checkpoint/vgg16.pth')  #
    print('model built')


    "Fashion - build the Evaluation plugin (Avalanche)"
    interactive_logger = InteractiveLogger()
    # TODO: Tensorboard Logger!
    if args.strategy == 'JT':
        eval_plugin = EvaluationPlugin(
            topk_acc_metrics(top_k=3, epoch=True, experience=True),
            topk_acc_metrics(top_k=5, epoch=True, experience=True),
            loggers=[interactive_logger]
        )
    else:
        eval_plugin = EvaluationPlugin(
            topk_acc_metrics(top_k=3, trained_experience=True),
            topk_acc_metrics(top_k=5, trained_experience=True),
            loggers=[interactive_logger]
        )


    " Fashion - CREATE THE STRATEGY INSTANCE (Replay)"
    if args.strategy == "CL":
        cl_strategy = Replay(
            model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
            CrossEntropyLoss(), mem_size=memory_size, device=device, train_mb_size=64, train_epochs=epochs, eval_mb_size=64,
            evaluator=eval_plugin)
    elif args.strategy == "Cum":
        cl_strategy = Cumulative(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
            CrossEntropyLoss(), device=device, train_mb_size=64, train_epochs=epochs, eval_mb_size=64,
            evaluator=eval_plugin)
    elif args.strategy == "JT":
        cl_strategy = JointTraining(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
                                   CrossEntropyLoss(), device=device, train_mb_size=64, train_epochs=epochs, eval_mb_size=64,
                                    evaluator=eval_plugin)
        scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=1, shuffle=True, seed=50, task_labels=False)
    else:
        ValueError("args.strategy must be (JT) (Cum) or (CL)!")

    "Print (DEBUG)"

    "Fashion - TRAINING LOOP"
    print('Starting experiment...')
    # list of dict/list to collect data
    results_list = []
    time_list = []
    for scenario in scenario_list:
        print("\n New RUN \n")
        results = []
        res = []
        timer.time = {}  # reset time!
        # reset model
        cl_strategy.model = GlobalCatePredictorFashion(num_classes=50, pretrained='checkpoint/vgg16.pth')  #

        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Number of  Pattern: ", len(experience.dataset))
            print("Current Classes: ", experience.classes_in_this_experience)

            timer.start()  #
            res.append(cl_strategy.train(experience, num_workers=4))
            timer.stop(experience.current_experience)  #
            print('Training completed')

            print('Computing accuracy on the whole test set')
            results.append(cl_strategy.eval(scenario.test_stream, num_workers=4))

        # Collect all the data
        results_list.append(results)
        time_list.append(timer.time)

    print()
    print("Final Results over 3 runs Eval:", results_list)
    print()
    print("Final Results over 3 run TR= ", time_list)
    print("DeepFashion - GPU n. ", cuda)


"""
RUN
"""
if __name__ == '__main__':
    main()
