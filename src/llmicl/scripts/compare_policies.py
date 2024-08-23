from typing import List
import argparse  # noqa: D100
from tqdm import tqdm
from pathlib import Path

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.linear_model import LinearRegression

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

import gymnasium as gym

from llmicl.interfaces import trainers
from llmicl.rl_helpers.rl_utils import create_env
# from llmicl.rl_helpers import nn_utils

from llmicl.rl_helpers.cleanrl_utils import SACActor, PPOAgent, make_env


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODE_LENGTH = 1000


DEFAULT_ENV_NAME: str = "HalfCheetah"
DEFAULT_TRIAL_NAME: str = "test"
DEFAULT_CONTEXT_LENGTH: int = 500
DEFAULT_VERBOSE: int = 0
DEFAULT_PREDICTION_HORIZON: int = 20
DEFAULT_START_FROM: int = 0
DEFAULT_USE_MC: bool = False
DEFAULT_N_EXPERIMENTS: int = 5
DEFAULT_POLICY_CHECKPOINT: List[int] = [0]
DEFAULT_TO_PLOT_MODELS: List[str] = []
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
    nargs="+",
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
    "--use_mc",
    metavar="use_mc",
    type=bool,
    help="if True, in addition to the llm, the estimated Markov Chain will be used for"
        "multi-step prediction",
    default=DEFAULT_USE_MC,
)
parser.add_argument(
    "--n_experiments",
    metavar="n_experiments",
    type=int,
    help="the number of experiments to conduct (number of episodes from the test "
        "dataset to consider)",
    default=DEFAULT_N_EXPERIMENTS,
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
print(f"---------- Env {args.env_name} ----------")
rescale_factor = 7.0
up_shift = 1.5

env, n_observations, n_actions = create_env(args.env_name)

columns = ["error", "model", "policy_checkpoint", "experiment"]
df = pd.DataFrame(columns=columns)
for i_checkpoint in args.policy_checkpoint:
    if 'sac' in args.policy_path:
        actor_builder = SACActor
    elif 'ppo' in args.policy_path:
        actor_builder = PPOAgent
    else:
        raise ValueError('RL agent not supported !')

    # ------------------------------ load policy ------------------------------
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_name, seed=7, idx=0, capture_video=False, run_name=""
            )
        ]
    )
    actor = actor_builder(envs).to(DEVICE)
    actor.load_state_dict(
        torch.load(f"{args.policy_path}_{i_checkpoint}.pth")
    )
    actor.eval()
    # -------------------------------------------------------------------------

    # ------------------------------ ICL ------------------------------
    all_mean_predictions = np.zeros(
        (args.context_length + args.prediction_horizon, n_observations)
    )

    # TODO: replace absolute paths with a variable
    if Path(
        "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
        f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
        f"llm_errors.npy"
    ).exists():
        llm_errors = np.load(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
            f"llm_errors.npy"
        )
        cst_errors = np.load(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
            f"cst_errors.npy"
        )
        if Path(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
            f"mc_errors.npy"
        ).exists():
            mc_errors = np.load(
                "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
                f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
                f"mc_errors.npy"
            )
    else:
        llm_errors = np.zeros(
            (args.context_length + args.prediction_horizon, args.n_experiments)
        )
        if args.use_mc:
            mc_errors = np.zeros(
                (args.context_length + args.prediction_horizon, args.n_experiments)
            )

        for i_exp in tqdm(range(args.n_experiments), desc="nbr of experiments"):
            # ------------------------------ generate time series ---------------------
            # rollout policy
            obs, _ = env.reset()
            restart = True
            df_lines = []
            for step in range(
                min(MAX_EPISODE_LENGTH, args.context_length+1+args.prediction_horizon)
            ):
                line = []
                line.append(obs[None, ...])
                if 'sac' in args.policy_path:
                    action, _, _ = actor.get_action(
                        torch.Tensor(obs[None, ...]).to(DEVICE)
                    )
                elif 'ppo' in args.policy_path:
                    action, _, _, _, _ = actor.get_action_and_value(
                        torch.Tensor(obs[None, ...]).to(DEVICE)
                    )
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
            # -------------------------------------------------------------------------

            # double-check for nan
            check_nan_with_actions = X[
                init_index : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                : n_observations + n_actions,
            ]
            if np.sum(np.isnan(check_nan_with_actions)) > 0:
                raise ValueError(
                    f"nan actions selected at indices "
                    f" {np.where(np.isnan(check_nan_with_actions))}"
                )

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
            trainer.icl(verbose=0)

            # groundtruth
            groundtruth = X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]

            if args.use_mc:
            # -------- Markov chain --------
                trainer.compute_statistics()
                trainer.build_tranistion_matrices(verbose=0)
                mc_predictions = trainer.predict_long_horizon_MC(
                    prediction_horizon=args.prediction_horizon,
                )
                mc_errors[:, i_exp] = np.linalg.norm(
                    groundtruth - mc_predictions,
                    axis=1,
                )

            # -------- LLM --------
            icl_object = trainer.predict_long_horizon_llm(
                prediction_horizon=args.prediction_horizon, stochastic=args.stochastic
            )

            llm_predictions = np.zeros(
                (args.context_length + args.prediction_horizon, n_observations)
            )
            for dim in range(n_observations):
                ts_max = icl_object[dim].rescaling_max
                ts_min = icl_object[dim].rescaling_min

                mode_arr = (
                    (icl_object[dim].mode_arr.flatten() - up_shift) / rescale_factor
                ) * (ts_max - ts_min) + ts_min
                all_mean_predictions[:, dim] = mode_arr

            llm_errors[:, i_exp] = np.linalg.norm(
                groundtruth - all_mean_predictions,
                axis=1,
            )

            # constant baseline
            cst_errors = np.zeros(
                (args.context_length + args.prediction_horizon, args.n_experiments)
            )
            cst_pred = copy.copy(
                X[
                    init_index : init_index + args.context_length + args.prediction_horizon,
                    :n_observations,
                ]
            )
            cst_pred[
                args.context_length : args.context_length + args.prediction_horizon
            ] = cst_pred[args.context_length - 1]

            # errors
            cst_errors[:, i_exp] = np.linalg.norm(
                groundtruth - cst_pred,
                axis=1,
            )

    # ---------------- save predictions ----------------
    models = ["cst", "llm"]
    if args.use_mc:
        models.append("mc")
    for m in models:
        np.save(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"env|{args.env_name}_policy|{i_checkpoint}_trial|{args.trial_name}_"
            f"{m}_errors.npy",
            eval(f"{m}_errors"),
        )

    """
    # train linear regression baseline and MLP
    X_baselines = load_offline_dataset(
        path=f"{args.data_path}/{args.env_name}/{dataset_name}/X_train.csv"
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

    # linreg model
    linreg_model = LinearRegression(fit_intercept=True)
    linreg_model.fit(X_train, y_train)
    # linreg model action
    linreg_model_actions = LinearRegression(fit_intercept=True)
    linreg_model_actions.fit(X_train_actions, y_train)
    # mlp
    mlp = nn_utils.NeuralNet(input_size=n_observations, output_size=n_observations)
    _, _, mlp = nn_utils.train_mlp(model=mlp, X_train=X_train, y_train=y_train)
    mlp.eval()
    # mlp + actions
    mlp_actions = nn_utils.NeuralNet(
        input_size=n_observations + n_actions, output_size=n_observations
    )
    _, _, mlp_actions = nn_utils.train_mlp(
        model=mlp_actions, X_train=X_train_actions, y_train=y_train
    )
    mlp_actions.eval()

    # initialize errors
    linreg_errors = np.zeros(
        (args.context_length + args.prediction_horizon, args.n_experiments)
    )
    linreg_actions_errors = np.zeros(
        (args.context_length + args.prediction_horizon, args.n_experiments)
    )
    mlp_errors = np.zeros(
        (args.context_length + args.prediction_horizon, args.n_experiments)
    )
    mlp_actions_errors = np.zeros(
        (args.context_length + args.prediction_horizon, args.n_experiments)
    )
    for i_exp in tqdm(range(args.n_experiments), desc="nbr of experiments"):
        init_index = episode_starts[i_exp]

        # groundtruth
        groundtruth = X[
            init_index + 1 : init_index
            + args.context_length
            + 1
            + args.prediction_horizon,
            :n_observations,
        ]

        # double-check for nan
        check_nan_with_actions = X[
            init_index : init_index
            + args.context_length
            + 1
            + args.prediction_horizon,
            : n_observations + n_actions,
        ]
        if np.sum(np.isnan(check_nan_with_actions)) > 0:
            raise ValueError(
                f"nan actions selected at indices "
                f" {np.where(np.isnan(check_nan_with_actions))}"
            )

        # linear actions
        linreg_input_actions = X[
            init_index : init_index + args.context_length + args.prediction_horizon,
            : n_observations + n_actions,
        ]
        linreg_pred_actions = linreg_model_actions.predict(linreg_input_actions)

        # linear
        linreg_input = X[
            init_index : init_index + args.context_length + args.prediction_horizon,
            :n_observations,
        ]
        linreg_pred = linreg_model.predict(linreg_input)

        # mlp
        mlp_pred = (
            mlp(torch.from_numpy(linreg_input).type(torch.FloatTensor))
            .cpu()
            .detach()
            .numpy()
        )

        # mlp + actions
        mlp_actions_pred = (
            mlp_actions(torch.from_numpy(linreg_input_actions).type(torch.FloatTensor))
            .cpu()
            .detach()
            .numpy()
        )

        # multi-step prediction
        for h in range(args.prediction_horizon):
            # linear + actions
            new_input_actions = np.concatenate(
                [
                    linreg_pred_actions[args.context_length + h - 1].reshape((1, -1)),
                    X[
                        init_index + args.context_length + h,
                        n_observations : n_observations + n_actions,
                    ].reshape((1, -1)),
                    # TODO: halfcheetah have obs_reward as additional obs
                ],
                axis=1,
            )
            new_pred_actions = linreg_model_actions.predict(new_input_actions)
            linreg_pred_actions[args.context_length + h] = copy.copy(new_pred_actions)
            # linear
            new_pred = linreg_model.predict(
                linreg_pred[args.context_length + h - 1].reshape((1, -1))
            )
            linreg_pred[args.context_length + h] = copy.copy(new_pred)
            # mlp
            new_pred = (
                mlp(
                    torch.from_numpy(
                        linreg_pred[args.context_length + h - 1].reshape((1, -1))
                    ).type(torch.FloatTensor)
                )
                .cpu()
                .detach()
                .numpy()
            )
            mlp_pred[args.context_length + h] = copy.copy(new_pred)
            # mlp actions
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

        linreg_errors[:, i_exp] = np.linalg.norm(
            groundtruth - linreg_pred,
            axis=1,
        )
        linreg_actions_errors[:, i_exp] = np.linalg.norm(
            groundtruth - linreg_pred_actions,
            axis=1,
        )
        mlp_errors[:, i_exp] = np.linalg.norm(
            groundtruth - mlp_pred,
            axis=1,
        )
        mlp_actions_errors[:, i_exp] = np.linalg.norm(
            groundtruth - mlp_actions_pred,
            axis=1,
        )

    # save predictions
    for m in ["linreg", "linreg_actions", "mlp", "mlp_actions"]:
        np.save(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"env|{args.env_name}_data|{dataset_name}_trial|{args.trial_name}_"
            f"{m}_errors.npy",
            eval(f"{m}_errors"),
        )
    """

    models += args.to_plot_models
    for i in range(llm_errors.shape[1]):
        for m in models:
            mini_df = pd.DataFrame(columns=columns)
            mini_df["error"] = eval(f"{m}_errors")[:, i]
            mini_df["model"] = m
            mini_df["policy_checkpoint"] = i_checkpoint
            mini_df["experiment"] = i
            df = pd.concat([df, mini_df], axis=0)
df = df.reset_index()
# ----------------------------------------------------------------------------------

# ------------------------------ Visualization ------------------------------
g = sns.relplot(
    data=df.loc[df["index"] > args.start_from],
    x="index",
    y="error",
    col="policy_checkpoint",
    hue="model",
    kind="line",  # palette="flare"
    # facet_kws=dict(legend_out=True)
)
plt.legend()

sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.5, 0.7),
    ncol=3,
    title=None,
    frameon=False,
)

for ax_idx, ax in enumerate(g.axes.flat):
    ax.axvline(x=args.context_length, color="black", linestyle=":")
    ax.grid(True)
    ax.set_ylabel("MSE")

g.figure.suptitle(
    f"Compare policies - Env: {args.env_name}"
)
plt.tight_layout()
plt.savefig(
    "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/compare_policies_"
    f"env|{args.env_name}_trial|{args.trial_name}.png"
)
plt.show()
# ----------------------------------------------------------------------------------
