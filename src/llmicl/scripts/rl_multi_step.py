import argparse  # noqa: D100

import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

import gymnasium as gym

from llmicl.interfaces import trainers
from llmicl.rl_helpers.rl_utils import load_offline_dataset, create_env


DEFAULT_ENV_NAME: str = "HalfCheetah"
DEFAULT_TRIAL_NAME: str = "test"
DEFAULT_DATA_LABEL: str = "expert"
DEFAULT_DATA_PATH: str = "/home/abenechehab/datasets"
DEFAULT_CONTEXT_LENGTH: int = 500
DEFAULT_INIT_INDEX: int = 0
DEFAULT_VERBOSE: int = 0
DEFAULT_PREDICTION_HORIZON: int = 20
DEFAULT_START_FROM: int = 0
DEFAULT_USE_LLM: bool = False


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
    "--episode",
    metavar="episode",
    type=int,
    help="the episode from which the 'context_length' transitions will be selected, has "
    "to be smaller than the number of episodes in the given dataset",
    default=DEFAULT_INIT_INDEX,
)
parser.add_argument(
    "--verbose",
    metavar="verbose",
    type=int,
    help="if 1, show progress bars for icl predictions",
    default=DEFAULT_VERBOSE,
)
parser.add_argument(
    "--prediction_horizon",
    metavar="prediction_horizon",
    type=int,
    help="the prediction horizon of the multi-step prediction",
    default=DEFAULT_PREDICTION_HORIZON,
)
parser.add_argument(
    "--start_from",
    metavar="start_from",
    type=int,
    help="the timestep (x-index) from which to start the plot",
    default=DEFAULT_START_FROM,
)
parser.add_argument(
    "--use_llm",
    metavar="use_llm",
    type=bool,
    help="if True, the llm will be used for multi-step prediction, otherwise it's the "
        "estimated Markov Chain (by multiplication)",
    default=DEFAULT_USE_LLM,
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

_, n_observations, n_actions = create_env(args.env_name)

# ------------------------------ generate time series ------------------------------
data_path = f"{args.data_path}/{args.env_name}/{args.data_label}/X_test.csv"
X = load_offline_dataset(path=data_path)

restart_index = n_observations+n_actions+1
restarts = X[:, restart_index]
episode_starts = np.where(restarts)[0]
assert args.episode <= len(
    episode_starts
), f"given episode {args.n_experiments} bigger than n episodes {len(episode_starts)}!!"

init_index = episode_starts[args.episode]

time_series = X[
    init_index : init_index + args.context_length, :n_observations
]
# ----------------------------------------------------------------------------------

# ------------------------------ ICL ------------------------------
rescale_factor = 7.0
up_shift = 1.5

env = gym.make(args.env_name)

trainer = trainers.RLICLTrainer(
    model=model,
    tokenizer=tokenizer,
    n_observations=n_observations,
    n_actions=n_actions,
    rescale_factor=rescale_factor,
    up_shift=up_shift,
)

trainer.update_context(
    time_series=copy.copy(time_series),
    mean_series=copy.copy(time_series),
    sigma_series=np.zeros_like(time_series),
    context_length=args.context_length,
    update_min_max=True,
)
trainer.icl(verbose=args.verbose)
# ---------------------------------------------------------------------------

# ------------------------------ Visualization ------------------------------
n_rows = (n_observations // 3) + 1
f, axes = plt.subplots(
    n_rows, 3, figsize=(20, 20), gridspec_kw={"wspace": 0.3}, sharex=True
)
axes = list(np.array(axes).flatten())
if args.use_llm:
    icl_object = trainer.predict_long_horizon_llm(
        prediction_horizon=args.prediction_horizon
    )

    for dim in range(n_observations):
        groundtruth = X[
            init_index + 1 : init_index
            + args.context_length
            + 1
            + args.prediction_horizon,
            dim,
        ].flatten()
        size_all = len(groundtruth)

        ts_max = icl_object[dim].rescaling_max
        ts_min = icl_object[dim].rescaling_min

        mode_arr = ((icl_object[dim].mode_arr.flatten() - up_shift) / rescale_factor) * (
            ts_max - ts_min
        ) + ts_min
        mean_arr = ((icl_object[dim].mean_arr.flatten() - up_shift) / rescale_factor) * (
            ts_max - ts_min
        ) + ts_min
        sigma_arr = (icl_object[dim].sigma_arr.flatten() / rescale_factor) * (
            ts_max - ts_min
        )

        x = np.arange(mean_arr.shape[0])
        axes[dim].plot(
            x[args.start_from :],
            mode_arr[args.start_from :],
            "k--",
            label="mode",
            alpha=0.5,
        )
        axes[dim].plot(
            x[args.start_from : -args.prediction_horizon],
            mean_arr[args.start_from : -args.prediction_horizon],
            label="mean +- std",
            color="blue",
            alpha=0.5,
        )
        axes[dim].fill_between(
            x=x[args.start_from : -args.prediction_horizon],
            y1=mean_arr[args.start_from : -args.prediction_horizon]
            - sigma_arr[args.start_from : -args.prediction_horizon],
            y2=mean_arr[args.start_from : -args.prediction_horizon]
            + sigma_arr[args.start_from : -args.prediction_horizon],
            alpha=0.15,
            color="blue",
        )
        axes[dim].plot(
            x[-args.prediction_horizon :],
            mean_arr[-args.prediction_horizon :],
            label="multi-step",
            color="orange",
        )
        axes[dim].fill_between(
            x=x[-args.prediction_horizon :],
            y1=mean_arr[-args.prediction_horizon :] - sigma_arr[-args.prediction_horizon :],
            y2=mean_arr[-args.prediction_horizon :] + sigma_arr[-args.prediction_horizon :],
            alpha=0.3,
            color="orange",
        )

        axes[dim].plot(
            x[args.start_from :],
            groundtruth[args.start_from :],
            label="gt",
            color="red",
            alpha=0.5,
        )
        axes[dim].set_title(f"{dim}")
        if dim >= ((n_observations // 3) * 3):
            axes[dim].set_xlabel("timesteps")
    axes[dim].legend()
    f.suptitle(
        f"Env {args.env_name}|{args.data_label} - multi-step prediction by llm",
        fontsize=16,
    )
    plt.savefig(
        "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/multi_step_llm_"
        f"env|{args.env_name}_data|{args.data_label}_ep|{args.episode}_trial"
        f"|{args.trial_name}.png"
    )
else:
    _ = trainer.compute_statistics()

    # MC kernel
    _, _ = trainer.build_tranistion_matrices(verbose=args.verbose)

    # multi-step prediction
    mc_predictions = trainer.predict_long_horizon_MC(
        prediction_horizon=args.prediction_horizon,
    )

    for dim in range(n_observations):
        groundtruth = X[
            args.init_index + 1 : args.init_index
            + args.context_length
            + 1
            + args.prediction_horizon,
            dim,
        ].flatten()
        size_all = len(groundtruth)

        x = np.arange(mc_predictions.shape[0])
        axes[dim].plot(
            x[args.start_from : -args.prediction_horizon],
            mc_predictions[args.start_from : -args.prediction_horizon, dim],
            label="mean +- std",
            color="blue",
            alpha=0.5,
        )
        axes[dim].plot(
            x[-args.prediction_horizon :],
            mc_predictions[-args.prediction_horizon :, dim],
            label="multi-step",
            color="orange",
        )
        axes[dim].plot(
            x[args.start_from :],
            groundtruth[args.start_from :],
            label="gt",
            color="red",
            alpha=0.5,
        )
        axes[dim].set_title(f"{dim}")
        if dim > 15:
            axes[dim].set_xlabel("timesteps")
    axes[dim].legend()
    f.suptitle(
        f"Env {args.env_name}|{args.data_label} - multi-step prediction by MC",
        fontsize=16
    )
    plt.savefig(
        "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/multi_step_mc_"
        f"{args.env_name}_{args.data_label}_{args.init_index}_{args.trial_name}.png"
    )
plt.show()
# ----------------------------------------------------------------------------------