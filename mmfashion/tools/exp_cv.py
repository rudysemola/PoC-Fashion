"""
    PoC Computer Vision (CORE50)
    Training (Avalanche) loop
"""

"script import"
import argparse

"Avalanche import"
from avalanche.benchmarks.classic import CORe50
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import MobilenetV1
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Cumulative, Replay


"Pytorch import"
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

"Python Files"
from utils_config import Timer


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
    # device
    cuda = int(args.cuda)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")
    # epochs
    epochs = int(args.epochs)
    # memory_size
    memory_size = int(args.memory_size)


    "CORE50 - Scenario and Benchmarck"
    scenario = CORe50(scenario="nc")

    "Fashion - build model"
    model = MobilenetV1(pretrained=True, latent_layer_num=20)
    print('model built')


    "CORE50 - build the Evaluation plugin (Avalanche)"
    interactive_logger = InteractiveLogger()
    # TODO: Tensorboard Logger!
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(trained_experience=True),
        accuracy_metrics(trained_experience=True),
        loggers=[interactive_logger]
    )


    " CORE50 - CREATE THE STRATEGY INSTANCE (Replay)"
    if args.strategy == "CL":
        cl_strategy = Replay(
            model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
            CrossEntropyLoss(), mem_size=memory_size, device=device, train_mb_size=128, train_epochs=epochs, eval_mb_size=64,
            evaluator=eval_plugin)
    elif args.strategy == "Cum":
        cl_strategy = Cumulative(model, SGD(model.parameters(), lr=1e-3, momentum=0.9),
            CrossEntropyLoss(), device=device, train_mb_size=128, train_epochs=epochs, eval_mb_size=64,
            evaluator=eval_plugin)
    else:
        ValueError("args.strategy must be (JT) (Cum) or (CL)!")

    "Print (DEBUG)"

    "CORE50 - TRAINING LOOP"
    print('Starting experiment...')
    results = []
    res = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Number of  Pattern: ", len(experience.dataset))
        print("Current Classes: ", experience.classes_in_this_experience)

        timer.start() #
        res.append(cl_strategy.train(experience, num_workers=4))
        timer.stop(experience.current_experience) #
        print('Training completed')

        print('Computing accuracy on the whole test set')
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
