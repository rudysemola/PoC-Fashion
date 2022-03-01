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
        accuracy_metrics(experience=True, stream=True),
        #timing_metrics(experience=True),
        loggers=[interactive_logger]
    )

    " Fashion - CREATE THE STRATEGY INSTANCE (Replay)" # total_epochs = 50
    # cl_strategy = Replay(
    #    model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
    #    CrossEntropyLoss(), mem_size=500, device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
    #    evaluator=eval_plugin)
    cl_strategy = Cumulative(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
        CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
        evaluator=eval_plugin)
    #cl_strategy = JointTraining(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
    #                           CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=1, eval_mb_size=64,
    #                            evaluator=eval_plugin)
    #scenario = nc_benchmark(train_dataset, val_dataset, n_experiences=1, shuffle=True, seed=10, task_labels=False)

    "Print (DEBUG)"
    """
    # Dataset
    print('Dataset len= ', len(dataset))
    print("TR len= ", len(train_dataset))
    print("VAL len= ", len(val_dataset))
    print("target TR example: ", train_dataset[1][1])
    print("img TR example: ", train_dataset[1][0])
    print("target VAL example: ", val_dataset[1][1])
    print()
    # Scenario
    print("scenario: ", scenario)
    train_stream = scenario.train_stream
    val_stream = scenario.test_stream
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



"""
RUN
"""
if __name__ == '__main__':
    main()
