import argparse  # noqa: D100
from tqdm import tqdm

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

import gymnasium as gym

from llmicl.interfaces import trainers
from llmicl.rl_helpers.rl_utils import load_offline_dataset
from llmicl.rl_helpers import nn_utils


DEFAULT_ENV_NAME: str = "HalfCheetah"
DEFAULT_TRIAL_NAME: str = "test"
DEFAULT_DATA_LABEL: str = "random"
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
    default=[DEFAULT_DATA_LABEL],
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
rescale_factor = 7.0
up_shift = 1.5
n_experiments = 19
training_data_size = 500

env = gym.make(args.env_name)
n_observations: int = env.observation_space.shape[0]
if len(env.action_space.shape) == 0:
    n_actions: int = 1
else:
    n_actions: int = env.action_space.shape[0]

columns = ["prediction", "model", "dataset", "episode"]
df = pd.DataFrame(columns=columns)
for dataset_name in args.data_label:
    # ------------------------------ generate time series ------------------------------
    data_path = f"{args.data_path}/{args.env_name}/{dataset_name}/X_test.csv"
    X = load_offline_dataset(path=data_path)
    # ----------------------------------------------------------------------------------

    # ------------------------------ ICL ------------------------------
    all_mean_predictions = np.zeros(
        (args.context_length + args.prediction_horizon, n_observations)
    )
    llm_errors = np.zeros((args.context_length + args.prediction_horizon, n_experiments))
    mc_errors = np.zeros((args.context_length + args.prediction_horizon, n_experiments))
    for i_exp in tqdm(range(n_experiments), desc="nbr of experiments"):
        init_index = int(i_exp * 1000)
        time_series = X[init_index : init_index + args.context_length, :n_observations]

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
            update_min_max=True,
        )
        trainer.icl(verbose=0)

        # -------- Markov chain --------
        trainer.compute_statistics()
        trainer.build_tranistion_matrices(verbose=0)
        mc_predictions = trainer.predict_long_horizon_MC(
            prediction_horizon=args.prediction_horizon,
        )
        mc_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - mc_predictions,
            axis=1,
        )

        # -------- LLM --------
        icl_object = trainer.predict_long_horizon_llm(
            prediction_horizon=args.prediction_horizon
        )

        llm_predictions = np.zeros(
            (args.context_length + args.prediction_horizon, n_observations)
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

            mode_arr = (
                (icl_object[dim].mode_arr.flatten() - up_shift) / rescale_factor
            ) * (ts_max - ts_min) + ts_min
            all_mean_predictions[:, dim] = mode_arr

        llm_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - all_mean_predictions,
            axis=1,
        )

    # train linear regression baseline and MLP
    X_baselines = load_offline_dataset(
        path=f"{args.data_path}/{args.env_name}/{dataset_name}/X_train.csv"
    )

    X_train_actions = copy.copy(
        np.concatenate(
            [
                X_baselines[:, :n_observations],
                X_baselines[:, n_observations + 1 : n_observations + 1 + n_actions],
            ],
            axis=1,
        )[:-1]
    )
    X_train = copy.copy(X_baselines[:-1, :n_observations])
    y_train = copy.copy(X_baselines[1:, :n_observations])

    nan_indices = np.unique(np.argwhere(np.isnan(X_train_actions))[:, 0])
    mask = np.ones(X_train_actions.shape[0], bool)
    mask[nan_indices] = False

    X_train = X_train[mask][:training_data_size]
    X_train_actions = X_train_actions[mask][:training_data_size]
    y_train = y_train[mask][:training_data_size]

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

    # constant and linreg baselines
    cst_errors = np.zeros(
        (args.context_length + args.prediction_horizon, n_experiments)
    )
    linreg_errors = np.zeros(
        (args.context_length + args.prediction_horizon, n_experiments)
    )
    linreg_actions_errors = np.zeros(
        (args.context_length + args.prediction_horizon, n_experiments)
    )
    mlp_errors = np.zeros(
        (args.context_length + args.prediction_horizon, n_experiments)
    )
    mlp_actions_errors = np.zeros(
        (args.context_length + args.prediction_horizon, n_experiments)
    )
    for i_exp in tqdm(range(n_experiments), desc="nbr of experiments"):
        init_index = int(i_exp * 1000)
        # constant
        cst_pred = copy.copy(
            X[
                init_index : init_index + args.context_length + args.prediction_horizon,
                :n_observations,
            ]
        )
        cst_pred[
            args.context_length : args.context_length + args.prediction_horizon
        ] = cst_pred[args.context_length - 1]
        cst_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - cst_pred,
            axis=1,
        )

        # linear actions
        linreg_input_actions = np.concatenate(
            [
                X[
                    init_index : init_index
                    + args.context_length
                    + args.prediction_horizon,
                    :n_observations,
                ],
                X[
                    init_index : init_index
                    + args.context_length
                    + args.prediction_horizon,
                    n_observations + 1 : n_observations + 1 + n_actions,
                ],
            ],
            axis=1,
        )
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
                        n_observations + 1 : n_observations + 1 + n_actions,
                    ].reshape((1, -1)),
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
                        n_observations + 1 : n_observations + 1 + n_actions,
                    ].reshape((1, -1)),
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
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - linreg_pred,
            axis=1,
        )
        linreg_actions_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - linreg_pred_actions,
            axis=1,
        )
        mlp_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - mlp_pred,
            axis=1,
        )
        mlp_actions_errors[:, i_exp] = np.linalg.norm(
            X[
                init_index + 1 : init_index
                + args.context_length
                + 1
                + args.prediction_horizon,
                :n_observations,
            ]
            - mlp_actions_pred,
            axis=1,
        )

    for i in range(llm_errors.shape[1]):
        # llm
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = llm_errors[:, i]
        mini_df["model"] = "llm"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # mc
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = mc_errors[:, i]
        mini_df["model"] = "mc"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # baseline:constant
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = cst_errors[:, i]
        mini_df["model"] = "baseline:constant"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # baseline:linear
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = linreg_errors[:, i]
        mini_df["model"] = "baseline:linear"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # baseline:linear+actions
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = linreg_actions_errors[:, i]
        mini_df["model"] = "baseline:linear+actions"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # baseline:mlp
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = mlp_errors[:, i]
        mini_df["model"] = "baseline:mlp"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)
        # baseline:mlp+actions
        mini_df = pd.DataFrame(columns=columns)
        mini_df["prediction"] = mlp_actions_errors[:, i]
        mini_df["model"] = "baseline:mlp+actions"
        mini_df["dataset"] = dataset_name
        mini_df["episode"] = i
        df = pd.concat([df, mini_df], axis=0)

    df = df.reset_index()

    # save predictions
    for model in ['llm', 'mc', 'linreg', 'linreg_actions', 'mlp', 'mlp_actions', 'cst']:
        np.save(
            "/home/abenechehab/llmicl/src/llmicl/artifacts/data/"
            f"{args.env_name}_{dataset_name}_{args.trial_name}_"
            f"{model}_errors.npy",
            eval(f"{model}_errors"),
        )
# ----------------------------------------------------------------------------------

# ------------------------------ Visualization ------------------------------
g = sns.relplot(
    data=df.loc[df["index"] > args.start_from],
    x="index",
    y="prediction",
    col="dataset",
    hue="model",
    kind="line",  # palette="flare"
    # facet_kws=dict(legend_out=True)
)

sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.5, 1),
    ncol=3,
    title=None,
    frameon=False,
)

for ax_idx, ax in enumerate(g.axes.flat):
    ax.axvline(x=args.context_length, color="black", linestyle=":")
    ax.grid(True)
    ax.set_ylabel("MSE")

g.fig.suptitle(
    f"Env: {args.env_name} | Training data size for baselines: {training_data_size}"
)
plt.tight_layout()
plt.savefig(
    "/home/abenechehab/llmicl/src/llmicl/artifacts/figures/baselines_vs_llm_"
    f"{args.env_name}_{args.trial_name}.png"
)
plt.show()
# ----------------------------------------------------------------------------------
