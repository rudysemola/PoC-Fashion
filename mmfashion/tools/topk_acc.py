################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 10-03-2022                                                             #
# Author(s): Rudy Semola                                                       #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Union, Dict

import torch
from torch import Tensor
# from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task

from collections import defaultdict


class TopkAccuracy(Metric[float]):
    """

    """

    def __init__(self, top_k):
        """
        """
        self._topk_acc_dict = defaultdict(Mean)
        # self.topk_metrics = Accuracy(top_k=top_k)
        self.top_k = top_k

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[float, Tensor],
    ) -> None:
        """
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        if isinstance(task_labels, Tensor) and len(task_labels) != len(true_y):
            raise ValueError("Size mismatch for true_y and task_labels tensors")

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if isinstance(task_labels, int):
            total_patterns = len(true_y)
            self._topk_acc_dict[task_labels].update(
                accuracy(predicted_y, true_y, top_k=self.top_k), total_patterns
            )
        elif isinstance(task_labels, Tensor):
            for pred, true, t in zip(predicted_y, true_y, task_labels):
                self._topk_acc_dict[t.item()].update(accuracy(pred, true, top_k=self.top_k), 1)
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )

    def result(self, task_label=None) -> Dict[int, float]:
        """
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v.result() for k, v in self._topk_acc_dict.items()}
        else:
            return {task_label: self._topk_acc_dict[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._topk_acc_dict = defaultdict(Mean)
        else:
            self._topk_acc_dict[task_label].reset()


class TopkAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    """

    def __init__(self, reset_at, emit_at, mode, top_k):
        self._topk_acc = TopkAccuracy(top_k=top_k)
        super(TopkAccuracyPluginMetric, self).__init__(
            self._topk_acc, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._topk_acc.update(strategy.mb_output, strategy.mb_y, task_labels)


class MinibatchTopkAccuracy(TopkAccuracyPluginMetric):
    """
    """

    def __init__(self, top_k):
        """
        """
        super(MinibatchTopkAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", top_k=top_k
        )

    def __str__(self):
        return "TopkAcc_MB"


class EpochTopkAccuracy(TopkAccuracyPluginMetric):
    """
    """

    def __init__(self, top_k):
        """
        """

        super(EpochTopkAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", top_k=top_k
        )

    def __str__(self):
        return "TopkAcc_Epoch"


class RunningEpochTopkAccurac(TopkAccuracyPluginMetric):
    """
    """

    def __init__(self, top_k):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochTopkAccurac, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", top_k=top_k
        )

    def __str__(self):
        return "TopkAcc_Running_Epoch"


class ExperienceTopkAccuracy(TopkAccuracyPluginMetric):
    """
    """

    def __init__(self, top_k):
        """
        """
        super(ExperienceTopkAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval", top_k=top_k
        )

    def __str__(self):
        return "TopkAcc_Exp"


class StreamTopkAccuracy(TopkAccuracyPluginMetric):
    """
    """

    def __init__(self, top_k):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamTopkAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", top_k=top_k
        )

    def __str__(self):
        return "TopkAcc_Stream"


def topk_acc_metrics(
    *,
    top_k=3,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
) -> List[PluginMetric]:
    """
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchTopkAccuracy(top_k=top_k))

    if epoch:
        metrics.append(EpochTopkAccuracy(top_k=top_k))

    if epoch_running:
        metrics.append(RunningEpochTopkAccurac(top_k=top_k))

    if experience:
        metrics.append(ExperienceTopkAccuracy(top_k=top_k))

    if stream:
        metrics.append(StreamTopkAccuracy(top_k=top_k))

    return metrics


# TODO
""""
__all__ = [
    "Accuracy",
    "MinibatchAccuracy",
    "EpochAccuracy",
    "RunningEpochAccuracy",
    "ExperienceAccuracy",
    "StreamAccuracy",
    "TrainedExperienceAccuracy",
    "accuracy_metrics",
]
"""


"""
UNIT TEST
"""
if __name__ == '__main__':
    print(topk_acc_metrics(experience=True, stream=True, top_k=3))
