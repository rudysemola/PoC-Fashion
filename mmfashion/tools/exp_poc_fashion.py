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
from utils_config import DatasetSetting, Timer
from global_cate_fashion_predictor import GlobalCatePredictorFashion
from topk_acc import *


"""
Function main
"""
def main():
    "Configs"
    # TIME
    timer = Timer()
    # dataset
    data_cfg = DatasetSetting()
    # GPU | Device
    ## (if multi-GPU -> 0 is Replay | 1 is Cumulative)
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
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print('dataset splitted')
    # build Benchmarks - Use nc_benchmark to setup the scenario and benchmark
    scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=10, shuffle=True, seed=50, task_labels=False, per_exp_classes={0:10})

    "Fashion - build model"
    model = GlobalCatePredictorFashion(num_classes=50, pretrained='checkpoint/vgg16.pth')  #
    print('model built')

    "Fashion - build the Evaluation plugin (Avalanche)"
    interactive_logger = InteractiveLogger()
    # TODO: Tensorboard Logger!
    eval_plugin = EvaluationPlugin(
        # TODO: TEST metrics 1 (return a metrics - see schema...)
        topk_acc_metrics(top_k=5, experience=True, stream=True),
        accuracy_metrics(experience=True, stream=True),
        #accuracy_metrics(epoch=True, experience=True, stream=True), # !!! ONLY FOR JOINT TR !!! epoch=True
        loggers=[interactive_logger]
    )

    " Fashion - CREATE THE STRATEGY INSTANCE (Replay)" # total_epochs = 50
    # TODO: TEST metrics 2
    cl_strategy = Replay(
        model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
        CrossEntropyLoss(), mem_size=10000, device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
        evaluator=eval_plugin)
    #cl_strategy = Cumulative(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
    #    CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=30, eval_mb_size=64,
    #    evaluator=eval_plugin)
    #cl_strategy = JointTraining(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
    #                           CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=40, eval_mb_size=64,
    #                            evaluator=eval_plugin)
    #scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=1, shuffle=True, seed=50, task_labels=False)

    "Print (DEBUG)"

    # TODO: (opz) test in inference (val-set) the metrics top-k in my code

    "Fashion - TRAINING LOOP"
    print('Starting experiment...')
    results = []
    res = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Number of  Pattern: ", len(experience.dataset))
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        timer.start() #
        res.append(cl_strategy.train(experience, num_workers=4))
        timer.stop(experience.current_experience) #
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # eval also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=4))

    print()
    print("Final Results eval:")
    print(results, "\n")
    print("Final Results TR= ")
    print(timer.time)
    print("GPU n. ", cuda)



"""
RUN
"""
if __name__ == '__main__':
    main()
