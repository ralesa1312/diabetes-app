import os

from ml.evaluate import evaluate
from ml.utils import DATA_DIR


def test_evaluate_generates_plot():
    plot_path = os.path.join(DATA_DIR, "metrics/evaluation_metrics.png")

    if os.path.exists(plot_path):
        os.remove(plot_path)

    evaluate()

    assert os.path.exists(plot_path)
