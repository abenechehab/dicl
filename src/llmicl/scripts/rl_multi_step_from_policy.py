import argparse  # noqa: D100
from typing import List

import copy
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

import gymnasium as gym

from llmicl.interfaces import trainers
from llmicl.rl_helpers.rl_utils import create_env
# from llmicl.rl_helpers import nn_utils

from llmicl.rl_helpers.cleanrl_utils import Actor, make_env


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODE_LENGTH = 1000

DEFAULT_ENV_NAME: str = "HalfCheetah"
DEFAULT_TRIAL_NAME: str = "test"
DEFAULT_CONTEXT_LENGTH: int = 500
DEFAULT_VERBOSE: int = 0
DEFAULT_PREDICTION_HORIZON: int = 20
DEFAULT_START_FROM: int = 0
DEFAULT_USE_LLM: bool = False
DEFAULT_TO_PLOT_MODELS: List[str] = []
DEFAULT_TRAINING_DATA_SIZE: int = 500
DEFAULT_POLICY_CHECKPOINT: int = 0
DEFAULT_STOCHASTIC: bool = False


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
    "--policy_path",
    metavar="policy_path",
    type=str,
    help="the folder that contains the policy checkpoints",
    required=True,
)
parser.add_argument(
    "--policy_checkpoint",
    metavar="policy_checkpoint",
    type=int,
    help="the policy checkpoint to load for rollout",
    default=DEFAULT_POLICY_CHECKPOINT,
)
parser.add_argument(
    "--context_length",
    metavar="context_length",
    type=int,
    help="the context length",
    default=DEFAULT_CONTEXT_LENGTH,
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
parser.add_argument(
    "--to_plot_models",
    metavar="to_plot_models",
    type=str,
    help="models to include in the plots",
    default=DEFAULT_TO_PLOT_MODELS,
    nargs="+",
)
parser.add_argument(
    "--training_data_size",
    metavar="training_data_size",
    type=int,
    help="the number of experiments to conduct (number of episodes from the test "
    "dataset to consider)",
    default=DEFAULT_TRAINING_DATA_SIZE,
)
parser.add_argument(
    "--stochastic",
    metavar="stochastic",
    type=bool,
    help="multi-step prediction done by sampling from the llm rather than taking the "
        "mode",
    default=DEFAULT_STOCHASTIC,
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

env, n_observations, n_actions = create_env(args.env_name)

# ------------------------------ generate time series ------------------------------
# load policy
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_name, seed=7, idx=0, capture_video=False, run_name="")]
)
actor = Actor(envs).to(DEVICE)
actor.load_state_dict(
    torch.load(f"{args.policy_path}/actor_checkpoint_{args.policy_checkpoint}.pth")
)
actor.eval()

# rollout policy
obs, _ = env.reset()
restart = True
df_lines = []
for step in range(
    min(MAX_EPISODE_LENGTH, args.context_length+1+args.prediction_horizon)
):
    line = []
    line.append(obs[None, ...])
    action, _, _ = actor.get_action(torch.Tensor(obs[None, ...]).to(DEVICE))
    action = action.detach().cpu().numpy().flatten()
    line.append(action[None, ...])
    obs_next, reward, terminated, truncated, _ = env.step(action)
    line.append(np.array([reward])[None, ...])
    line.append(np.array([int(restart)])[None, ...])
    line = np.concatenate(line, axis=1)
    df_lines.append(line)
    restart = False
    obs = obs_next
    if terminated or truncated:
        break

X = np.concatenate(df_lines, axis=0)

init_index = 0

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
"""
# ------------------------------ Baselines ------------------------------
# train linear regression baseline and MLP
X_baselines = load_offline_dataset(
    path=f"{args.data_path}/{args.env_name}/{args.data_label}/X_train.csv"
)

X_train_actions = copy.copy(
    np.concatenate(
        [
            X_baselines[:, :n_observations],
            X_baselines[:, n_observations : n_observations + n_actions],
        ],
        axis=1,
    )[:-1]
)  # TODO: halfcheetah have obs_reward as additional obs
X_train = copy.copy(X_baselines[:-1, :n_observations])
y_train = copy.copy(X_baselines[1:, :n_observations])

nan_indices = np.unique(np.argwhere(np.isnan(X_train_actions))[:, 0])
mask = np.ones(X_train_actions.shape[0], bool)
mask[nan_indices] = False

X_train = X_train[mask][: args.training_data_size]
X_train_actions = X_train_actions[mask][: args.training_data_size]
y_train = y_train[mask][: args.training_data_size]

if 'linreg' in args.to_plot_models:
    linreg_model = LinearRegression(fit_intercept=True)
    linreg_model.fit(X_train, y_train)
    # multi step prediction
    linreg_input = X[
        init_index : init_index + args.context_length + args.prediction_horizon,
        :n_observations,
    ]
    linreg_pred = linreg_model.predict(linreg_input)
    # multi-step prediction
    for h in range(args.prediction_horizon):
        new_pred = linreg_model.predict(
            linreg_pred[args.context_length + h - 1].reshape((1, -1))
        )
        linreg_pred[args.context_length + h] = copy.copy(new_pred)
if 'linreg_actions' in args.to_plot_models:
    linreg_model_actions = LinearRegression(fit_intercept=True)
    linreg_model_actions.fit(X_train_actions, y_train)
    # multi step prediction
    linreg_input_actions = X[
        init_index : init_index + args.context_length + args.prediction_horizon,
        : n_observations + n_actions,
    ]
    linreg_actions_pred = linreg_model_actions.predict(linreg_input_actions)
    # multi-step prediction
    for h in range(args.prediction_horizon):
        # linear + actions
        new_input_actions = np.concatenate(
            [
                linreg_actions_pred[args.context_length + h - 1].reshape((1, -1)),
                X[
                    init_index + args.context_length + h,
                    n_observations : n_observations + n_actions,
                ].reshape((1, -1)),
                # TODO: halfcheetah have obs_reward as additional obs
            ],
            axis=1,
        )
        new_pred_actions = linreg_model_actions.predict(new_input_actions)
        linreg_actions_pred[args.context_length + h] = copy.copy(new_pred_actions)
if 'mlp' in args.to_plot_models:
    mlp = nn_utils.NeuralNet(input_size=n_observations, output_size=n_observations)
    _, _, mlp = nn_utils.train_mlp(model=mlp, X_train=X_train, y_train=y_train)
    mlp.eval()
    # multi step prediction
    mlp_input = X[
        init_index : init_index + args.context_length + args.prediction_horizon,
        :n_observations,
    ]
    mlp_pred = (
        mlp(torch.from_numpy(mlp_input).type(torch.FloatTensor)).cpu().detach().numpy()
    )
    # multi-step prediction
    for h in range(args.prediction_horizon):
        new_pred = (
            mlp(
                torch.from_numpy(
                    mlp_pred[args.context_length + h - 1].reshape((1, -1))
                ).type(torch.FloatTensor)
            )
            .cpu()
            .detach()
            .numpy()
        )
        mlp_pred[args.context_length + h] = copy.copy(new_pred)
if 'mlp_actions' in args.to_plot_models:
    mlp_actions = nn_utils.NeuralNet(
        input_size=n_observations + n_actions, output_size=n_observations
    )
    _, _, mlp_actions = nn_utils.train_mlp(
        model=mlp_actions, X_train=X_train_actions, y_train=y_train
    )
    mlp_actions.eval()
    # multi step prediction
    mlp_input_actions = X[
        init_index : init_index + args.context_length + args.prediction_horizon,
        : n_observations + n_actions,
    ]
    mlp_actions_pred = (
        mlp_actions(torch.from_numpy(mlp_input_actions).type(torch.FloatTensor))
        .cpu()
        .detach()
        .numpy()
    )
    # multi-step prediction
    for h in range(args.prediction_horizon):
        new_input_actions = np.concatenate(
            [
                mlp_actions_pred[args.context_length + h - 1].reshape((1, -1)),
                X[
                    init_index + args.context_length + h,
                    n_observations : n_observations + n_actions,
                ].reshape((1, -1)),
                # TODO: halfcheetah have obs_reward as additional obs
            ],
            axis=1,
        )
        new_pred_actions = (
            mlp_actions(torch.from_numpy(new_input_actions).type(torch.FloatTensor))
            .cpu()
            .detach()
            .numpy()
        )
        mlp_actions_pred[args.context_length + h] = copy.copy(new_pred_actions)
# ---------------------------------------------------------------------------
"""
# ------------------------------ Visualization ------------------------------
n_rows = (n_observations // 3) + 1
f, axes = plt.subplots(
    n_rows, 3, figsize=(20, 20), gridspec_kw={"wspace": 0.3}, sharex=True
)
axes = list(np.array(axes).flatten())
if args.use_llm:
    icl_object = trainer.predict_long_horizon_llm(
        prediction_horizon=args.prediction_horizon,
        stochastic=args.stochastic,
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
            x[-args.prediction_horizon - 1 :],
            mean_arr[-args.prediction_horizon - 1 :],
            label="multi-step",
            color="orange",
        )
        axes[dim].fill_between(
            x=x[-args.prediction_horizon - 1 :],
            y1=mean_arr[-args.prediction_horizon - 1 :]
            - sigma_arr[-args.prediction_horizon - 1 :],
            y2=mean_arr[-args.prediction_horizon - 1 :]
            + sigma_arr[-args.prediction_horizon - 1 :],
            alpha=0.3,
            color="orange",
        )

        # -----------------
        initial_state = X[init_index + args.context_length, dim]
        """
        if 'linreg' in args.to_plot_models:
            axes[dim].plot(
                x[-args.prediction_horizon - 1 :],
                np.concatenate(
                    [
                        initial_state.reshape((1,)),
                        linreg_pred[-args.prediction_horizon :, dim],
                    ],
                    axis=0,
                ),
                label="linreg",
                color="green",
            )
        if 'linreg_actions' in args.to_plot_models:
            axes[dim].plot(
                x[-args.prediction_horizon - 1 :],
                np.concatenate(
                    [
                        initial_state.reshape((1,)),
                        linreg_actions_pred[-args.prediction_horizon :, dim],
                    ],
                    axis=0,
                ),
                label="linreg_actions",
                color="brown",
            )
        if 'mlp' in args.to_plot_models:
            axes[dim].plot(
                x[-args.prediction_horizon - 1 :],
                np.concatenate(
                    [
                        initial_state.reshape((1,)),
                        mlp_pred[-args.prediction_horizon :, dim],
                    ],
                    axis=0,
                ),
                label="mlp",
                color="cyan",
            )
        if 'mlp_actions' in args.to_plot_models:
            axes[dim].plot(
                x[-args.prediction_horizon - 1 :],
                np.concatenate(
                    [
                        initial_state.reshape((1,)),
                        mlp_actions_pred[-args.prediction_horizon :, dim],
                    ],
                    axis=0,
                ),
                label="mlp_actions",
                color="purple",
            )
        # -----------------
        """

        axes[dim].plot(
            x[-args.prediction_horizon-1 :],
            initial_state * np.ones_like(x[-args.prediction_horizon-1 :]),
            label="constant",
            color="gray",
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
        f"Env {args.env_name}|Policy {args.policy_checkpoint} - multi-step prediction by llm",
        fontsize=16,
    )
    plt.savefig(
        "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/multi_step_llm_"
        f"env|{args.env_name}_policy|{args.policy_checkpoint}_trial"
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
f.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------