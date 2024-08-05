from typing import Callable, Dict

import evaluate
import numpy as np
from transformers.trainer_utils import EvalPrediction


def get_metrics_function() -> Callable[[EvalPrediction], Dict[str, float]]:
    """Get metrics function for evaluation of accuracy and Macro F1 score.

    Returns:
        Callable[[EvalPrediction], Dict[str, float]: Metrics function
    """
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics_function(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred

        # Take argmax to get the predicted class
        predictions = np.argmax(logits, axis=-1)

        # Compute accuracy and F1 score
        metrics = metric_accuracy.compute(predictions=predictions, references=labels)
        metrics.update(metric_f1.compute(predictions=predictions, references=labels, average="macro"))
        return metrics

    return compute_metrics_function
