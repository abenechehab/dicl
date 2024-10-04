from llmicl.interfaces import trainers

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from transformers import LlamaForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING

import copy
import numpy as np
import pandas as pd

import gymnasium as gym

import torch

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM, AutoTokenizer
# --------------------------------------------

state_names = {
    "HalfCheetah": [
        "rootz",
        "rooty",
        "bthigh",
        "bshin",
        "bfoot",
        "fthigh",
        "fshin",
        "ffoot",
        "rootx_dot",
        "rootz_dot",
        "rooty_dot",
        "bthigh_dot",
        "bshin_dot",
        "bfoot_dot",
        "fthigh_dot",
        "fshin_dot",
        "ffoot_dot",
    ],
    "Hopper": [
        "rootz",
        "rooty",
        "thigh",
        "leg",
        "foot",
        "rootx_dot",
        "rootz_dot",
        "rooty_dot",
        "thigh_dot",
        "leg_dot",
        "foot_dot",
    ],
    "Ant": [
        "rootz",
        "root_quat_x",
        "root_quat_y",
        "root_quat_z",
        "root_quat_w",
        "hip_1_angle",
        "ankle_1_angle",
        "hip_2_angle",
        "ankle_2_angle",
        "hip_3_angle",
        "ankle_3_angle",
        "hip_4_angle",
        "ankle_4_angle",
        "rootx_dot",
        "rootz_dot",
        "rooty_dot",
        "root_quat_x_dot",
        "root_quat_y_dot",
        "root_quat_z_dot",
        "hip_1_angle_dot",
        "ankle_1_angle_dot",
        "hip_2_angle_dot",
        "ankle_2_angle_dot",
        "hip_3_angle_dot",
        "ankle_3_angle_dot",
        "hip_4_angle_dot",
        "ankle_4_angle_dot",
    ],
    "Walker2d": [
        "rootz",
        "rooty",
        "thigh_right_angle",
        "leg_right_angle",
        "foot_right_angle",
        "thigh_left_angle",
        "leg_left_angle",
        "foot_left_angle",
        "rootx_dot",
        "rootz_dot",
        "rooty_dot",
        "thigh_right_angle_dot",
        "leg_right_angle_dot",
        "foot_right_angle_dot",
        "thigh_left_angle_dot",
        "leg_left_angle_dot",
        "foot_left_angle_dot",
    ],
    "Swimmer": [
        "rootx",
        "rootz",
        "rooty",
        "joint_1_angle",
        "joint_2_angle",
        "rootx_dot",
        "rootz_dot",
        "rooty_dot",
        "joint_1_angle_dot",
        "joint_2_angle_dot",
    ],
}
action_names = {
    "HalfCheetah": [
        "t_bthigh",
        "t_bshin",
        "t_bfoot",
        "t_fthigh",
        "t_fshin",
        "t_ffoot",
    ],
    "Hopper": [
        "thigh_joint",
        "leg_joint",
        "foot_joint",
    ],
    "Ant": [
        "hip_4_joint",
        "ankle_4_joint",
        "hip_1_joint",
        "ankle_1_joint",
        "hip_2_joint",
        "ankle_2_joint",
        "hip_3_joint",
        "ankle_3_joint",
    ],
    "Walker2d": [
        "thigh_left_joint",
        "leg_left_joint",
        "foot_left_joint",
        "thigh_right_joint",
        "leg_right_joint",
        "foot_right_joint",
    ],
    "Swimmer": [
        "joint_1",
        "joint_2",
    ],
}


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self, input_array, y=None):
        return input_array * 1


def invert_reduction(
    icl_object,
    reduction_object,
    scaling_pipeline,
    n_observations,
    n_components,
    original_series,
    rescale_factor: float = 7.0,
    up_shift: float = 1.5,
    use_scaler: bool = True,
):
    groundtruth = scaling_pipeline.transform(
        original_series
    )
    scaled_mean_error = []
    if use_scaler:
        predictions = scaling_pipeline.inverse_transform(
            reduction_object.inverse_transform(
                np.concatenate(
                    [
                        icl_object[dim].predictions[..., None]
                        for dim in range(n_components)
                    ],
                    axis=1,
                )
            )
        )[:, :n_observations]
    else:
        predictions = reduction_object.inverse_transform(
            np.concatenate(
                [icl_object[dim].predictions[..., None] for dim in range(n_components)],
                axis=1,
            )
        )
    all_mean = []
    all_mode = []
    all_lb = []
    all_ub = []
    for dim in range(n_components):
        ts_max = icl_object[dim].rescaling_max
        ts_min = icl_object[dim].rescaling_min
        # # -------------------- Useful for Plots --------------------
        mode_arr = (
            (icl_object[dim].mode_arr.flatten() - up_shift) / rescale_factor
        ) * (ts_max - ts_min) + ts_min
        mean_arr = (
            (icl_object[dim].mean_arr.flatten() - up_shift) / rescale_factor
        ) * (ts_max - ts_min) + ts_min
        sigma_arr = (icl_object[dim].sigma_arr.flatten() / rescale_factor) * (
            ts_max - ts_min
        )

        all_mean.append(mean_arr[..., None])
        all_mode.append(mode_arr[..., None])
        all_lb.append(mean_arr[..., None] - sigma_arr[..., None])
        all_ub.append(mean_arr[..., None] + sigma_arr[..., None])

    if use_scaler:
        all_mean = scaling_pipeline.inverse_transform(
            reduction_object.inverse_transform(np.concatenate(all_mean, axis=1))
        )
        all_mode = scaling_pipeline.inverse_transform(
            reduction_object.inverse_transform(np.concatenate(all_mode, axis=1))
        )
        all_lb = scaling_pipeline.inverse_transform(
            reduction_object.inverse_transform(np.concatenate(all_lb, axis=1))
        )
        all_ub = scaling_pipeline.inverse_transform(
            reduction_object.inverse_transform(np.concatenate(all_ub, axis=1))
        )
    else:
        all_mean = reduction_object.inverse_transform(np.concatenate(all_mean, axis=1))
        all_mode = reduction_object.inverse_transform(np.concatenate(all_mode, axis=1))
        all_lb = reduction_object.inverse_transform(np.concatenate(all_lb, axis=1))
        all_ub = reduction_object.inverse_transform(np.concatenate(all_ub, axis=1))

    scaled_mean_error = np.abs(all_mean[-100:] - groundtruth[-100:]).mean(axis=0)

    return predictions, all_mean, all_mode, all_lb, all_ub, scaled_mean_error


def full_pipeline(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    env_name: str,
    data_label: str,
    episode: int,
    method: str,
    label: str,
    states_and_actions: bool = False,
    n_components: int = -1,
    prediction_horizon: int = 20,
    rescale_factor: float = 7.0,
    up_shift: float = 1.5,
    only_use_context: bool = False,
    context_length: int = 500,
    verbose: int = 0
):
    env = gym.make(env_name)
    n_actions = env.action_space.shape[0]
    n_observations = env.observation_space.shape[0]

    # load some data to get a pool of states
    data_path = f"/home/abenechehab/datasets/{env_name}/{data_label}/X_test.csv"
    X = pd.read_csv(data_path, index_col=0)
    X = X.values.astype('float')

    # find episodes beginnings
    restart_index = n_observations+n_actions
    restarts = X[:, restart_index+1]
    episode_starts = np.where(restarts)[0]

    init_index = episode_starts[episode]

    n_original_components = (
        n_observations + n_actions if states_and_actions else n_observations
    )
    if not only_use_context:
        obs = X[:, :n_original_components]
    else:
        obs = X[init_index : init_index + context_length, :n_original_components]
    obs = obs[~np.isnan(obs[:,-1])]

    # standard scaling
    scaling_pipeline = make_pipeline(MinMaxScaler(), StandardScaler())
    scaling_pipeline.fit(obs)
    obs_scaled = scaling_pipeline.transform(obs)

    n_components = n_components if n_components != -1 else n_original_components

    if method == 'vanilla':
        reduction_object = IdentityTransformer()
    elif method == 'pca':
        reduction_object = PCA(n_components=n_components)
    elif method == 'ica':
        reduction_object = FastICA(n_components=n_components)
    else:
        raise ValueError(f'method [{method}] is not supported!')

    reduction_object.fit(obs_scaled)

    time_series = reduction_object.transform(
        scaling_pipeline.transform(
            X[init_index : init_index + context_length, :n_original_components]
        )
    )

    # ------------ ICL -----------
    trainer = trainers.RLICLTrainer(
        model=model,
        tokenizer=tokenizer,
        n_observations=n_components,
        n_actions=n_actions,
        rescale_factor=rescale_factor,
        up_shift=up_shift,
    )

    trainer.update_context(
        time_series=copy.copy(time_series),
        mean_series=copy.copy(time_series),
        sigma_series=np.zeros_like(time_series),
        context_length=context_length,
        update_min_max=True,
    )
    trainer.icl(verbose=verbose, stochastic=True)

    if prediction_horizon > 0:
        icl_object = trainer.predict_long_horizon_llm(
            prediction_horizon=prediction_horizon,
            stochastic=True,
            verbose=verbose
        )
    else:
        icl_object = trainer.compute_statistics()

    predictions, all_mean, all_mode, all_lb, all_ub, scaled_mean_error = (
        invert_reduction(
            icl_object=icl_object,
            reduction_object=reduction_object,
            scaling_pipeline=scaling_pipeline,
            n_observations=n_observations,
            n_components=n_components,
            original_series=X[
                init_index + 1 : init_index + context_length + 1, :n_original_components
            ],
            rescale_factor=rescale_factor,
            up_shift=up_shift,
            use_scaler=True,
        )
    )

    if states_and_actions:
        scaled_all_mean = scaling_pipeline.transform(all_mean)[:,:n_observations]
        scaled_all_mode = scaling_pipeline.transform(all_mode)[:, :n_observations]
        groundtruth = X[
            init_index + 1 : init_index + context_length + prediction_horizon + 1,
            :n_observations+n_actions
        ]
        scaled_groundtruth = scaling_pipeline.transform(groundtruth)[:, :n_observations]
    else:
        scaled_all_mean = scaling_pipeline.transform(all_mean)
        scaled_all_mode = scaling_pipeline.transform(all_mode)
        groundtruth = X[
            init_index + 1 : init_index + context_length + prediction_horizon + 1,
            :n_observations
        ]
        scaled_groundtruth = scaling_pipeline.transform(groundtruth)

    columns = [
        'env_name',
        'data_label',
        'episode',
        'method',
        'states_and_actions',
        'label',
        'n_components',
        'mode_or_mean',
        'prediction_horizon',
    ]

    columns += state_names[env_name]

    fixed_part = [
        env_name,
        data_label,
        episode,
        method,
        states_and_actions,
        label,
        n_components,
    ]
    df_lines = []
    for h in range(context_length+prediction_horizon):
        # mode
        new_line = copy.copy(fixed_part) + ['mode', h+1]
        new_line += (
            np.abs(scaled_groundtruth[h, :] - scaled_all_mode[h, :]).flatten().tolist()
        )
        df_lines.append(pd.DataFrame([new_line], columns=columns))
        # mean
        new_line = copy.copy(fixed_part) + ['mean', h+1]
        new_line += (
            np.abs(scaled_groundtruth[h, :] - scaled_all_mean[h, :]).flatten().tolist()
        )
        df_lines.append(pd.DataFrame([new_line], columns=columns))
    return pd.concat(df_lines, axis=0), all_mean, all_mode, all_lb, all_ub, groundtruth


def full_pipeline_policy(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    env_name: str,
    data_label: str,
    episode: int,
    method: str,
    label: str,
    states_and_actions: bool = False,
    n_components: int = -1,
    prediction_horizon: int = 20,
    rescale_factor: float = 7.0,
    up_shift: float = 1.5,
    only_use_context: bool = False,
    context_length: int = 500,
):
    env = gym.make(env_name)
    n_actions = env.action_space.shape[0]
    n_observations = env.observation_space.shape[0]

    # load some data to get a pool of states
    data_path = f"/home/abenechehab/datasets/{env_name}/{data_label}/X_test.csv"
    X = pd.read_csv(data_path, index_col=0)
    X = X.values.astype("float")

    # find episodes beginnings
    restart_index = n_observations + n_actions  # +1 for halfcheetah
    restart_index += 1 if env_name == "HalfCheetah" else 0
    restarts = X[:, restart_index + 1]
    episode_starts = np.where(restarts)[0]

    init_index = episode_starts[episode]

    n_original_components = (
        n_observations + n_actions if states_and_actions else n_observations
    )
    if not only_use_context:
        obs = X[:, :n_original_components]
    else:
        obs = X[init_index : init_index + context_length, :n_original_components]
    obs = obs[~np.isnan(obs[:, -1])]

    # standard scaling
    scaling_pipeline = make_pipeline(MinMaxScaler(), StandardScaler())
    scaling_pipeline.fit(obs)
    obs_scaled = scaling_pipeline.transform(obs)

    n_components = n_components if n_components != -1 else n_original_components

    if method == "vanilla":
        reduction_object = IdentityTransformer()
    elif method == "pca":
        reduction_object = PCA(n_components=n_components)
    elif method == "ica":
        reduction_object = FastICA(n_components=n_components)
    else:
        raise ValueError(f"method [{method}] is not supported!")

    reduction_object.fit(obs_scaled)

    time_series = reduction_object.transform(
        scaling_pipeline.transform(
            X[init_index : init_index + context_length, :n_original_components]
        )
    )

    # ------------ ICL -----------
    trainer = trainers.RLICLTrainer(
        model=model,
        tokenizer=tokenizer,
        n_observations=n_components,
        n_actions=n_actions,
        rescale_factor=rescale_factor,
        up_shift=up_shift,
    )

    trainer.update_context(
        time_series=copy.copy(time_series),
        mean_series=copy.copy(time_series),
        sigma_series=np.zeros_like(time_series),
        context_length=context_length,
        update_min_max=True,
    )
    trainer.icl(verbose=1, stochastic=True)

    if prediction_horizon > 0:
        icl_object = trainer.predict_long_horizon_llm(
            prediction_horizon=prediction_horizon, stochastic=True, verbose=1
        )
    else:
        icl_object = trainer.compute_statistics()

    predictions, all_mean, all_mode, all_lb, all_ub, scaled_mean_error = (
        invert_reduction(
            icl_object=icl_object,
            reduction_object=reduction_object,
            scaling_pipeline=scaling_pipeline,
            n_observations=n_observations,
            n_components=n_components,
            original_series=X[
                init_index + 1 : init_index + context_length + 1, :n_components
            ],
            rescale_factor=rescale_factor,
            up_shift=up_shift,
            use_scaler=True,
        )
    )
    all_mean = scaling_pipeline.transform(all_mean)
    all_mode = scaling_pipeline.transform(all_mode)

    groundtruth = X[
        init_index + 1 : init_index + context_length + prediction_horizon + 1,
        :n_observations,
    ]
    groundtruth = scaling_pipeline.transform(groundtruth)

    columns = [
        "env_name",
        "data_label",
        "episode",
        "method",
        "states_and_actions",
        "label",
        "n_components",
        "mode_or_mean",
        "prediction_horizon",
    ]

    columns += state_names[env_name]

    fixed_part = [
        env_name,
        data_label,
        episode,
        method,
        states_and_actions,
        label,
        n_components,
    ]
    df_lines = []
    for h in range(context_length + prediction_horizon):
        # mode
        new_line = copy.copy(fixed_part) + ["mode", h + 1]
        new_line += np.abs(groundtruth[h, :] - all_mode[h, :]).flatten().tolist()
        df_lines.append(pd.DataFrame([new_line], columns=columns))
        # mean
        new_line = copy.copy(fixed_part) + ["mean", h + 1]
        new_line += np.abs(groundtruth[h, :] - all_mean[h, :]).flatten().tolist()
        df_lines.append(pd.DataFrame([new_line], columns=columns))
    return pd.concat(df_lines, axis=0)
