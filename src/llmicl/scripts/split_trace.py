import argparse
import pathlib
from tqdm import tqdm

import numpy as np
import pandas as pd


DEFAULT_SIZE = 100000
DEFAULT_TRAIN_PERCENTAGE = 80
DEFAULT_MIN_EPISODE_SIZE = 100


def preprocess_time(data: pd.DataFrame, timestamp_name: str = ""):
    """Preprocess time.

    Creates a timestamp if absent in metadata and sorts values by time.
    """
    if timestamp_name == "":
        timestamp_name = "fake_ts"
        data[timestamp_name] = data.index

    data[timestamp_name] = pd.to_datetime(data[timestamp_name])
    data.sort_values(by=[timestamp_name], inplace=True)
    data.set_index([timestamp_name], inplace=True)

    return data


parser = argparse.ArgumentParser(
    description="Split trace.csv into training and test datasets"
)
parser.add_argument(
    "--dataset",
    metavar="dataset",
    type=str,
    help="the name of the folder that contains trace.csv, must be inside data/",
)
parser.add_argument(
    "--size",
    metavar="size",
    type=int,
    help="the size of the subset to split",
    default=DEFAULT_SIZE,
)
parser.add_argument(
    "--train_percentage",
    metavar="train_percentage",
    type=int,
    help="[between 0 and 100] the percentage of the training set "
        "(of size percentage*size)",
    default=DEFAULT_TRAIN_PERCENTAGE,
)
parser.add_argument(
    "--min_episode_size",
    metavar="min_episode_size",
    type=int,
    help="The mnimum size for an episode to be considered",
    default=DEFAULT_MIN_EPISODE_SIZE,
)


args = parser.parse_args()
assert (args.train_percentage < 100) and (
    args.train_percentage > 0
), "train_percentage needs to be strictly between 0 and 100."
f" {args.train_percentage} was given"


trace_filename = "trace.csv"
output_dir = pathlib.Path(args.dataset)
trace_path = output_dir / trace_filename

print(f"start split for {args.dataset}")

trace_df = pd.read_csv(trace_path)
trace_df = preprocess_time(trace_df)

# number of samples to use in the training set
restart_name = "restart"
restarts = trace_df[restart_name].to_numpy()
episode_starts = np.where(restarts)[0]

training_size = args.size * args.train_percentage / 100

test_indices = []
train_indices = []

# when we dont count an episode, we add its size here
decount_steps = 0

for i, start in tqdm(
    enumerate(episode_starts[:-1]), desc="episodes", total=len(episode_starts)
):
    if start - decount_steps > args.size:
        break
    if (episode_starts[i + 1] - start < args.min_episode_size):
        decount_steps += episode_starts[i + 1] - start
        pass
    else:
        if start - decount_steps > training_size:
            test_indices += list(np.arange(start, episode_starts[i + 1]))
        else:
            train_indices += list(np.arange(start, episode_starts[i + 1]))

train_trace_df = trace_df.iloc[train_indices]
test_trace_df = trace_df.iloc[test_indices]

train_trace_df.to_csv(str(output_dir / "X_train.csv"))
test_trace_df.to_csv(str(output_dir / "X_test.csv"))

print("done !")
