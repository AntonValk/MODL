from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorboard as tb
import os

from tensorboard.backend.event_processing import event_accumulator


def summarize_experiments(dataset, filter_path = "lightning_logs/new"):
    filter_path = filter_path + "/" + dataset
    result = []
    for f in os.listdir(filter_path):
        ea = event_accumulator.EventAccumulator(filter_path + "/" + f)
        ea.Reload()
        result.append(ea.Scalars("train/cumulative_error_step")[-1].value)
    result = np.array(result)
    return result.mean(), result.std()
