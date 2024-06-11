### Set up directory
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
from pathlib import Path
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

import numpy as np
import pandas as pd

import copy
from tqdm import tqdm

import torch
from ICL import MultiResolutionPDF

from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler

import gymnasium as gym

from data.serialize import serialize_arr, deserialize_str, SerializerSettings
# -------------------------------------------------------------------------

def calculate_multiPDF_llama3(
    full_series, model, tokenizer, temperature=1.0, number_of_tokens_original=None,
):
    '''
     This function calculates the multi-resolution probability density function (PDF) for a given series.

     Parameters:
     full_series (str): The series for which the PDF is to be calculated.
     prec (int): The precision of the PDF.
     mode (str, optional): The mode of calculation. Defaults to 'neighbor'.
     refine_depth (int, optional): The depth of refinement for the PDF. Defaults to 1.
     llama_size (str, optional): The size of the llama model. Defaults to '13b'.
        
     Returns:
     list: A list of PDFs for the series.
    '''
    good_tokens_str = []
    for num in range(1000):
        good_tokens_str.append(str(num))
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    batch = tokenizer(
        [full_series],
        return_tensors="pt",
        add_special_tokens=True        
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(batch['input_ids'].cuda())  # use_cache=True)
    
    logit_mat = out['logits']
    
    # kv_cache_main = out['past_key_values']
    logit_mat_good = logit_mat[:,:,good_tokens].clone()
    
    if number_of_tokens_original:
        probs = torch.nn.functional.softmax(logit_mat_good[:,-(number_of_tokens_original-1):,:] / temperature, dim=-1)
    else:
        probs = torch.nn.functional.softmax(logit_mat_good[:,1:,:] / temperature, dim=-1)
    
    
    PDF_list = []
    
    # start_loop_from = 1 if use_instruct else 0
    for i in tqdm(range(1,int(probs.shape[1]),2)):
        PDF = MultiResolutionPDF()
        PDF.bin_center_arr = np.arange(0,1000) / 100
        PDF.bin_width_arr = np.array(1000*[0.01])
        PDF.bin_height_arr = probs[0,i,:].cpu().numpy() * 100
        PDF_list.append(PDF)
    
    # release memory
    del logit_mat  #, kv_cache_main
    return PDF_list
    

def serialize_gaussian(prec, time_series, mean_series, sigma_series):
    """
    Serialize a time series with gaussian noise and continuous support.

    Parameters:
    prec (int): Precision of the serialization
    time_series (list): The time series data
    mean_series (list): The mean series data
    sigma_series (list): The sigma series data

    Returns:
    tuple: A tuple containing 
        serialized time series: str
        rescaled mean series: np array
        rescaled sigma series: np array
    """
    settings=SerializerSettings(base=10, prec=prec, signed=True, time_sep=',', bit_sep='', minus_sign='-', fixed_length=False, max_val = 10)
    time_series = np.array(time_series)
    ### Final range is from 0.15 to 0.85
    rescale_factor = 7.0
    up_shift = 1.5

    rescaled_array = (time_series-time_series.min())/(time_series.max()-time_series.min()) * rescale_factor + up_shift
    rescaled_true_mean_arr = (np.array(mean_series)-time_series.min())/(time_series.max()-time_series.min()) * rescale_factor + up_shift
    rescaled_true_sigma_arr = np.array(sigma_series)/(time_series.max()-time_series.min()) * rescale_factor 
    # rescaled_true_mean_arr *= 10
    # rescaled_true_sigma_arr *= 10
    full_series = serialize_arr(rescaled_array, settings)
    return (full_series, rescaled_true_mean_arr, rescaled_true_sigma_arr)


def gym_generate_random_policy(Number_of_steps: int = 200, env_name: str = 'HalfCheetah', seed: int = 7):
    env = gym.make(env_name)
    N_observations = env.observation_space.shape[0]
    N_actions = env.action_space.shape[0]
    
    env.np_random.__setstate__(np.random.default_rng(seed).__getstate__())
    
    obs, _ = env.reset()
    
    observations = []
    actions = []
    rewards = []
    terminateds = []
    truncateds = []
    # Generate the episode
    for t in range(Number_of_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)
        obs = copy.copy(next_obs)
    
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0).reshape((-1, 1))
    
    X = np.concatenate([observations, actions, rewards], axis=1)
    return X, N_observations, N_actions

def load_offline_dataset(path: str):
    X = pd.read_csv(path, index_col=0)
    N_observations = 17
    N_actions = 6
    return X.values.astype('float'), X.columns, N_observations, N_actions

def make_RL_time_serie(
    X: np.array, 
    N_observations: int, 
    N_actions: int,
    Number_of_steps: int = 200,
    all_dim: bool = False, 
    dim: int = 8, 
    add_actions: bool = False, 
    add_reward: bool = False,
    traj_starting_idx: int = 0,
):
    N_dim = 0
    
    if all_dim:
        episode_time_series = X[traj_starting_idx:traj_starting_idx+Number_of_steps,:N_observations]
        N_dim += N_observations
    else:
        episode_time_series = X[traj_starting_idx:traj_starting_idx+Number_of_steps,dim].reshape((-1,1))
        N_dim += 1
    
    time_series_gt = episode_time_series  #[:, obs_dim]
    
    # concatenate obs time serie with actions at each timestep
    if add_actions:
        time_series_gt = np.concatenate(
            [time_series_gt, X[traj_starting_idx:traj_starting_idx+Number_of_steps,N_observations:N_observations+N_actions]], # +1 for the reward
            axis=1
        )
        N_dim += N_actions
    if add_reward:
        time_series_gt = np.concatenate(
            [time_series_gt, X[traj_starting_idx:traj_starting_idx+Number_of_steps,N_observations+N_actions:N_observations+N_actions+1]], # +1 for the reward
            axis=1
        )
        N_dim += 1
    
    # Rescale all the dimensions to [0,1]
    print(f"sample before rescale: {time_series_gt[:1,:1]}")
    scaler = MinMaxScaler()
    scaler.fit(time_series_gt)
    time_series_gt = scaler.transform(time_series_gt)
    print(f"sample after rescale: {time_series_gt[:1,:1]}")
    
    time_series = time_series_gt.flatten()
    
    mean_series = copy.copy(time_series)
    std_series = np.zeros_like(mean_series)
    
    full_series, rescaled_true_mean_arr, rescaled_true_sigma_arr = serialize_gaussian(2, time_series, mean_series, std_series)
    
    print(f"full_series: {full_series[:10]}")
    
    # Save the generated data to a dictionary
    series_dict = {
        'full_series': full_series,
        'rescaled_true_mean_arr': rescaled_true_mean_arr,
        'rescaled_true_sigma_arr': rescaled_true_sigma_arr,
        'time_series': np.array(time_series)
    }

    return series_dict, N_dim

def icl_prediction(model, tokenizer, series_dict: dict, temperature: float, pre_prompt: str = ""):
    T = 1.0
    number_of_tokens_original = None
    pre_prompt = ""
    full_series = series_dict['full_series']
    # print(f"full_series: {full_series[:10]}")
    # number_of_tokens_original = len(tokenizer(full_series)['input_ids']) 
    # print(f"number_of_tokens_original: {number_of_tokens_original}")
    PDF_list = calculate_multiPDF_llama3(
        pre_prompt+full_series,
        model=model,
        tokenizer=tokenizer,
        temperature=T,
        number_of_tokens_original=number_of_tokens_original,
    )
    series_dict['PDF_list'] = PDF_list
    return series_dict

def compute_statistics(series_dict: dict):
    full_series = series_dict['full_series']
    PDF_list = series_dict['PDF_list']
    rescaled_true_mean_arr = series_dict['rescaled_true_mean_arr']
    rescaled_true_sigma_arr = series_dict['rescaled_true_sigma_arr']
    
    PDF_true_list = copy.deepcopy(PDF_list)
    discrete_BT_loss = []
    discrete_KL_loss = []
    def cdf(x):
        return 0.5 * (1 + erf((x - true_mean) / (true_sigma * np.sqrt(2))))
    
    ### Extract statistics from MultiResolutionPDF
    mean_arr = []
    mode_arr = []
    sigma_arr = []
    moment_3_arr = []
    moment_4_arr = []
    for PDF, PDF_true, true_mean, true_sigma in zip(PDF_list, PDF_true_list, rescaled_true_mean_arr, rescaled_true_sigma_arr):
        PDF_true.discretize(cdf, mode = "cdf")
        PDF_true.compute_stats()
        discrete_BT_loss += [PDF_true.BT_dist(PDF)]    
        discrete_KL_loss += [PDF_true.KL_div(PDF)]
        
        PDF.compute_stats()
        mean, mode, sigma = PDF.mean, PDF.mode, PDF.sigma 
        moment_3 = PDF.compute_moment(3)
        moment_4 = PDF.compute_moment(4)
        
        mean_arr.append(mean)
        mode_arr.append(mode)
        sigma_arr.append(sigma)
        moment_3_arr.append(moment_3)
        moment_4_arr.append(moment_4)

    kurtosis_arr = np.array(moment_4_arr) / np.array(sigma_arr)**4
    statistics = {
        'mean_arr': np.array(mean_arr),
        'mode_arr': np.array(mode_arr),
        'sigma_arr': np.array(sigma_arr),
        'moment_3_arr': np.array(moment_3_arr),
        'moment_4_arr': np.array(moment_4_arr),
        'kurtosis_arr': kurtosis_arr,
        'kurtosis_error': (kurtosis_arr-3)**2,
        'error_mean': np.abs(rescaled_true_mean_arr - mean_arr),
        'error_mode': np.abs(rescaled_true_mean_arr - mode_arr),
        'discrete_BT_loss': np.array(discrete_BT_loss),
        'discrete_KL_loss': np.array(discrete_KL_loss),
    }
    return statistics

def to_plot_stats(statistics: dict, series_dict: dict, N_dim: int, Number_of_steps: int = 200):
    mode_arr = statistics['mode_arr'][N_dim-1:-1].reshape((-1, N_dim)) 
    mean_arr = statistics['mean_arr'][N_dim-1:-1].reshape((-1, N_dim))
    sigma_arr = statistics['sigma_arr'][N_dim-1:-1].reshape((-1, N_dim))
    print(f"mode_arr shape: {mode_arr.shape}")
    gt = series_dict['rescaled_true_mean_arr'].reshape((Number_of_steps, -1))[1:,:]
    print(f"time_series_gt shape: {gt.shape}")
    return gt, mode_arr, mean_arr, sigma_arr