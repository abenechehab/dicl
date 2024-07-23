import argparse  # noqa: D100

import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

import gymnasium as gym

from llmicl.interfaces import trainers
from llmicl.rl_helpers.rl_utils import load_offline_dataset


DEFAULT_ENV_NAME: str = "HalfCheetah"
DEFAULT_TRIAL_NAME: str = "test"
DEFAULT_DATA_LABEL: str = "d4rl_expert"
DEFAULT_DATA_PATH: str = "/home/abenechehab/d4rl"
DEFAULT_CONTEXT_LENGTH: int = 500
DEFAULT_INIT_INDEX: int = 0
DEFAULT_VERBOSE: int = 0


# -------------------- Parse arguments --------------------
parser = argparse.ArgumentParser(
    description="Split trace.csv into training and test datasets",
)
parser.add_argument(
    "--env_name",
    metavar="env_name",
    type=str,
    help="the environment name, must be inside envs/",
    default=DEFAULT_ENV_NAME,
)
parser.add_argument(
    "--trial_name",
    metavar="trial_name",
    type=str,
    help="the trial name, must be inside experiments/'env_name'/trials",
    default=DEFAULT_TRIAL_NAME,
)
parser.add_argument(
    "--data_path",
    metavar="data_path",
    type=str,
    help="the folder that contains the data, must contain this: "
        "env_name/data_label/X.test.csv",
    default=DEFAULT_DATA_PATH,
)
parser.add_argument(
    "--data_label",
    metavar="data_label",
    type=str,
    help="the name of the folder that contains trace.csv, must be inside data/",
    default=DEFAULT_DATA_LABEL,
)
parser.add_argument(
    "--context_length",
    metavar="context_length",
    type=int,
    help="the context length",
    default=DEFAULT_CONTEXT_LENGTH,
)
parser.add_argument(
    "--init_index",
    metavar="init_index",
    type=int,
    help="the index from which the 'context_length' transitions will be selected, has "
        "to be smaller than 'episode length - context_length'",
    default=DEFAULT_INIT_INDEX,
)
parser.add_argument(
    "--verbose",
    metavar="verbose",
    type=int,
    help="if 1, show progress bars for icl predictions",
    default=DEFAULT_VERBOSE,
)

args = parser.parse_args()
# ----------------------------------------

# ------------------------------ load model and tokenizer --------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "/home/gpaolo/nas_2/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/"
        "62bd457b6fe961a42a631306577e622c83876cb6/",
    use_fast=False,
)
print("finish loading tokenizer")
model = LlamaForCausalLM.from_pretrained(
    "/home/gpaolo/nas_2/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/"
        "62bd457b6fe961a42a631306577e622c83876cb6/",
    device_map="auto",
    torch_dtype=torch.float16,
)
print("finish loading model")
model.eval()
# ----------------------------------------------------------------------------------

# ------------------------------ generate time series ------------------------------
data_path = f"{args.data_path}/{args.env_name}/{args.data_label}/X_test.csv"
X, _, n_observations, n_actions = load_offline_dataset(path=data_path)

time_series = X[
    args.init_index : args.init_index + args.context_length, :n_observations
]
# ----------------------------------------------------------------------------------

# ------------------------------ ICL ------------------------------
rescale_factor = 7.0
up_shift = 1.5

env = gym.make(args.env_name)

trainer = trainers.RLICLTrainer(
    env=env,
    model=model,
    tokenizer=tokenizer,
    rescale_factor=rescale_factor,
    up_shift=up_shift,
)

trainer.update_context(
    time_series=copy.copy(time_series),
    mean_series=copy.copy(time_series),
    sigma_series=np.zeros_like(time_series),
    context_length=args.context_length,
)
trainer.icl(verbose=args.verbose)
icl_object = trainer.compute_statistics()
# ----------------------------------------------------------------------------------

# ------------------------------ Visualization ------------------------------
n_rows = (n_observations // 3) + 1
f, axes = plt.subplots(
    n_rows, 3, figsize=(20, 20), gridspec_kw={"wspace": 0.3}, sharex=True
)
axes = list(np.array(axes).flatten())
for dim in range(n_observations):
    groundtruth = X[
        args.init_index + 1 : args.init_index + args.context_length + 1, dim
    ].flatten()

    ts_max = icl_object[dim].rescaling_max
    ts_min = icl_object[dim].rescaling_min
    # -------------------- Useful for Plots --------------------
    mode_arr = ((icl_object[dim].mode_arr.flatten() - up_shift) / rescale_factor) * (
        ts_max - ts_min
    ) + ts_min
    mean_arr = ((icl_object[dim].mean_arr.flatten() - up_shift) / rescale_factor) * (
        ts_max - ts_min
    ) + ts_min
    sigma_arr = (icl_object[dim].sigma_arr.flatten() / rescale_factor) * (
        ts_max - ts_min
    )

    # -------------------- Trajectory visualization --------------------
    x = np.arange(mean_arr.shape[0])

    axes[dim].plot(x, mode_arr, "k--", label="mode")
    axes[dim].plot(x, mean_arr, label="mean +- std", color="blue")
    axes[dim].fill_between(x=x, y1=mean_arr - sigma_arr, y2=mean_arr + sigma_arr)
    axes[dim].plot(x, groundtruth, label="gt", color="red")
    axes[dim].set_title(f"{dim}")
    if dim >= ((n_observations // 3) * 3):
        axes[dim].set_xlabel("timesteps")
axes[dim].legend()
plt.savefig(
    "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/"
        f"{args.env_name}_{args.data_label}_{args.trial_name}_{args.init_index}.png"
)
plt.show()
# ----------------------------------------------------------------------------------