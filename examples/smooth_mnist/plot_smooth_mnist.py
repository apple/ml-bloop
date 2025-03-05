#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import argparse
from typing import NamedTuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class PlotConfig(NamedTuple):
    label: str
    color: str
    marker: str


PLOT_CONFIGS = {
    "mixed": PlotConfig(label="Mixed", color="blue", marker="o"),
    "bloop": PlotConfig(label="BLOOP", color="orange", marker="*"),
    "pcgrad": PlotConfig(label="PCGrad", color="green", marker="x"),
    "dynamic_barrier": PlotConfig(label="Dynamic Barrier", color="red", marker="v"),
}


METRIC_NAMES = {
    "train_loss": "Train loss",
    "train_accuracy": "Train accuracy",
    "test_loss": "Test loss",
    "test_accuracy": "Test accuracy",
    "aux_loss": "Lip. upper bound",
}


def get_method_name(filename):
    splitted = filename.split("_")
    method = splitted[-1]
    if method == "barrier":
        method = "dynamic_barrier"
    return method


def plot_pareto_front(
    metric_dict: dict[str, dict[str, np.ndarray]],
    x_metric: str,
    y_metric: str,
    save_dir: str,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
):
    metrics_to_plot = [x_metric, y_metric]

    # For legend
    seen_methods = set()

    for filename, metrics in metric_dict.items():
        method_name = get_method_name(filename)
        if method_name in seen_methods:
            label = None
        else:
            seen_methods.add(method_name)
            label = PLOT_CONFIGS[method_name].label

        color = PLOT_CONFIGS[method_name].color
        marker = PLOT_CONFIGS[method_name].marker
        plt.plot(
            *[metrics[metric] for metric in metrics_to_plot],
            color=color,
            alpha=0.1,
        )
        plt.scatter(
            *[metrics[metric][-1] for metric in metrics_to_plot],
            color=color,
            marker=marker,
            label=label,
            s=200,
        )
    plt.xlabel(METRIC_NAMES[x_metric])
    plt.ylabel(METRIC_NAMES[y_metric])
    if x_log_scale:
        plt.xscale("log")
    if y_log_scale:
        plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    filename = os.path.join(save_dir, f"pareto_{x_metric}_{y_metric}.png")
    plt.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot scripts to visualize Pareto front")
    parser.add_argument(
        "--metric_dir",
        type=str,
        default="results/metrics",
        help="Directory containing metrics to plot.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/figures",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    metric_dict = dict()

    for run in os.listdir(args.metric_dir):
        metrics = dict(np.load(os.path.join(args.metric_dir, run)))
        # The loss is the logarithm of the upper bound
        metrics["aux_loss"] = np.exp(metrics["aux_loss"])
        metric_dict[os.path.splitext(run)[0]] = metrics

    matplotlib.rcParams.update({"font.size": 24})
    x_metrics = ["train_loss", "test_loss"]

    for x_metric in x_metrics:
        y_metric = "aux_loss"
        # plot pareto front
        plt.figure(figsize=(12, 8))
        plot_pareto_front(
            metric_dict=metric_dict,
            x_metric=x_metric,
            y_metric=y_metric,
            save_dir=args.save_dir,
            x_log_scale=True,
        )
        plt.close()
