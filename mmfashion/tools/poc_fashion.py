"""
    PoC Fashion (Category prediction)
    Training (Avalanche) loop
"""

"mmFashion import"
from mmfashion.models import build_predictor
from mmfashion.utils import init_weights_from

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

    # build Benchmarks - Use nc_benchmark to setup the scenario and benchmark
    scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=5, shuffle=True, seed=1234, task_labels=False)

    "Fashion - build model"
    # build model
    model = GlobalCatePredictorFashion(num_classes=50, pretrained='checkpoint/vgg16.pth') #
    print('model built')

    "Fashion - build the Evaluation plugin (Avalanche)"
    # print to stdout
    interactive_logger = InteractiveLogger()
    # TODO: Tensorboard Logger!
    eval_plugin = EvaluationPlugin(
        #accuracy_metrics(epoch=True, experience=True, stream=True),
        accuracy_metrics(experience=True, stream=True),
        #timing_metrics(epoch=True, experience=True),
        timing_metrics(experience=True),
        loggers=[interactive_logger]
    )

    " Fashion - CREATE THE STRATEGY INSTANCE (Replay)"
    #cl_strategy = Replay(
    #    model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
    #    CrossEntropyLoss(), mem_size=500, device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
    #    evaluator=eval_plugin)
    cl_strategy = Cumulative(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
        CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
        evaluator=eval_plugin)
    #total_epochs = 50
    #work_dir = 'checkpoint/CateAttrPredict/vgg/global'

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

    print("target example: ", train_dataset[1][1])
    print("Inference model: ", model(train_dataset[1][0].unsqueeze(0)).shape)
    print("Inference model: ", model(train_dataset[1][0].unsqueeze(0)).squeeze())

    # Print Model test (DEBUG)
    #print(model)
    #print("scenario.n_classes= ", scenario.n_classes)
    #print()

    # Print CL strategy (DEBUG)
    #print(cl_strategy)
    #print()

    "Fashion - TRAINING LOOP"
    print('Starting experiment...')
    results = []
    res = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Number of  Pattern: ", len(experience.dataset))
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        res.append(cl_strategy.train(experience, num_workers=4))
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # eval also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=4))

    print("Final Results eval:")
    print(results, "\n")
    print("Final Results TR= ")
    print(res)



"""
RUN
"""
if __name__ == '__main__':
    main()
