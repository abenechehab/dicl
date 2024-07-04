from typing import List

import numpy as np
import matplotlib.pyplot as plt

import copy

### Set up directory
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)


import gymnasium as gym


def gym_generate_deterministic_policy(env: gym.Env, policy: np.array, Number_of_steps: int = 200, seed: int = 7):

    env.np_random.__setstate__(np.random.default_rng(seed).__getstate__())

    obs, _ = env.reset()

    observations = []
    actions = []
    rewards = []
    terminateds = []
    truncateds = []
    # Generate the episode
    for t in range(Number_of_steps):
        action = policy[obs]
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(np.array(obs).reshape(1,-1))
        actions.append(np.array(action).reshape(1,-1))
        rewards.append(np.array(reward).reshape(1,-1))
        terminateds.append(np.array(terminated).reshape(1,-1))
        truncateds.append(np.array(truncated).reshape(1,-1))
        
        if terminated or truncated:
            observations.append(np.array(next_obs).reshape(1,-1))
            none_action = np.empty(np.array(action).reshape(1,-1).shape)
            none_action[:] = np.nan
            actions.append(none_action)
            none_reward = np.empty(np.array(reward).reshape(1,-1).shape)
            none_reward[:] = np.nan
            rewards.append(none_reward)
            none_terminated = np.empty(np.array(terminated).reshape(1,-1).shape)
            none_terminated[:] = np.nan
            terminateds.append(none_terminated)
            none_truncated = np.empty(np.array(truncated).reshape(1,-1).shape)
            none_truncated[:] = np.nan
            truncateds.append(none_truncated)
            obs, _ = env.reset()
        else:
            obs = copy.copy(next_obs)

    observations = np.stack(observations, axis=0).reshape((-1, 1))
    actions = np.stack(actions, axis=0).reshape((-1, 1))
    rewards = np.stack(rewards, axis=0).reshape((-1, 1))

    X = np.concatenate([observations, actions, rewards], axis=1)
    return X


def make_RL_time_serie_discrete(
    X: np.array,
    Number_of_steps: int = 200,
    traj_starting_idx: int = 0,
    add_actions: bool = False,
):
    if add_actions:
        episode_time_series = X[traj_starting_idx:traj_starting_idx+Number_of_steps,:2]
    else:
        episode_time_series = X[traj_starting_idx:traj_starting_idx+Number_of_steps,:1]
    time_series = episode_time_series.flatten()
    full_series = ",".join(str(int(x)) if not np.isnan(x) else 'done' for x in time_series)
    print(f"full_series: {full_series[:25]}")
    # Save the generated data to a dictionary
    series_dict = {
        'full_series': full_series,
        'actions': X[traj_starting_idx:traj_starting_idx+Number_of_steps,1:2].flatten()
    }
    return series_dict

def build_transition_matrix(P_true: dict):
    n_states = len(P_true.keys())
    n_actions = len(P_true[0].keys())
    P = np.zeros((n_states, n_actions, n_states))
    for state in range(n_states):
        for action in range(n_actions):
            for transition in P_true[state][action]:
                proba, next_state, _, _ = transition
                P[state, action, next_state] = proba
    return P

def build_baseline_transition_matrix(n_states: int, n_actions: int, states: np.array, actions: np.array, add_actions: bool = False):
    P_baseline = (1/n_states) * np.ones((n_states, n_actions, n_states))
    visited = np.zeros((n_states, n_actions))
    for i in range(len(actions)-1):
        action = actions[i]
        if not np.isnan(action):
            state_index = i if not add_actions else 2*i
            state = int(states[state_index])
            if visited[state, int(action)]==0.0:
                P_baseline[state][int(action)] *= 0.0
            P_baseline[state][int(action)][int(states[i+1 if not add_actions else 2+2*i])] += 1.0
            visited[state, int(action)] = 1.0
    line_sums = np.sum(P_baseline, axis=2)
    non_zero_sums_mask = line_sums <= 1e-3
    new_line_sums = np.where(non_zero_sums_mask, 1.0, line_sums)
    P_baseline /= new_line_sums[...,None]
    return P_baseline

def kl_div(p: np.array,q: np.array):
    KL = np.sum(p * (np.log(p) - np.log(q)))  # check dim
    return KL

def compute_statistics_discrete(series_dict, add_actions: bool = False):
    series = series_dict['full_series'].split(',')
    actions = series_dict['actions']
    probs = series_dict['probs']

    P = series_dict['true_P']
    print(f"true probability matrix P: {P.shape}")

    N_states, N_actions, _ = P.shape
    P_hat = (1/N_states) * np.ones_like(P)

    visited = np.zeros((N_states, N_actions))
    
    mode_arr = []
    discrete_BT_loss = []
    discrete_KL_loss = []
    P_diff = []
    
    for i in range(len(actions)):
        action = actions[i]
        state_index = i if not add_actions else 2*i
        probs_index = 2*i if not add_actions else 2+4*i
        state = int(series[state_index])
        if not np.isnan(action):
            true_proba = P[state][int(action)].flatten()
            predicted_proba = probs[0, probs_index].flatten()
            p_line = P_hat[state][int(action)]
            if visited[state, int(action)]==0.0:
                P_hat[state][int(action)] = copy.copy(predicted_proba)
            else:
                P_hat[state][int(action)] = (p_line + copy.copy(predicted_proba)) / 2
            visited[state, int(action)] = 1.0

            P_diff.append(np.linalg.norm(P-P_hat))
            
            # KL divergence
            discrete_KL_loss.append(kl_div(p=true_proba,q=predicted_proba))
            
            # Bhattacharyya distance
            sqrt_PQ = np.sum(np.sqrt(predicted_proba * true_proba))
            BH = -np.log(sqrt_PQ)
            discrete_BT_loss.append(BH)

            # mode_prediction
            mode_arr.append(np.argmax(predicted_proba))

    series_dict['predicted_P'] = P_hat
            
    # loss_array = np.zeros(full_array[start_idx:].squeeze().shape)
    # # for state in states[[0,1]]:
    # for state in states:
    #     ### add 2 to slice out BOS
    #     pos = np.where(full_array[start_idx:]==state)[0]+2
    #     learned_p_out = probs.squeeze()[start_idx:][pos]
    #     true_p_out = torch.tensor(P[state])
    #     if dist_type == 'KL':
    #     ### KL divergence at each row
    #         KL = torch.sum(true_p_out * (true_p_out.log() - learned_p_out.log()), dim = 1)
    #         dist = KL
    #     else:
    #         ### Bhattacharyya distance at each row
    #         sqrt_PQ = torch.sum(torch.sqrt(learned_p_out * true_p_out), dim = 1)
    #         BH_dist = -torch.log(sqrt_PQ)
    #         dist = BH_dist
        
    #     loss_array[pos-2] = dist.numpy()

    statistics = {
        'mode_arr': np.array(mode_arr),
        'discrete_BT_loss': discrete_BT_loss,
        'discrete_KL_loss': discrete_KL_loss,
        'P_diff': np.array(P_diff)
    }

    return series_dict, statistics

def compute_statistics_discrete_policy(series_dict, policy, add_actions: bool = False):
    series = series_dict['full_series'].split(',')
    actions = series_dict['actions']
    probs = series_dict['probs']

    P = series_dict['true_P']
    print(f"true probability matrix P: {P.shape}")

    N_states, N_actions, _ = P.shape
    
    P_hat = (1/N_states) * np.ones_like(P)
    
    predicted_P_pi = (1/N_states) * np.ones((N_states, N_states))
    true_P_pi = (1/N_states) * np.ones((N_states, N_states))

    visited = np.zeros((N_states, N_actions))
    
    mode_arr = []
    discrete_BT_loss = []
    discrete_KL_loss = []
    P_diff = []
    P_pi_diff = []
    
    for i in range(len(actions)):
        action = actions[i]
        state_index = i if not add_actions else 2*i
        probs_index = 2*i if not add_actions else 2+4*i
        state = int(series[state_index])
        if not np.isnan(action):
            true_proba = P[state][int(action)].flatten()
            predicted_proba = probs[0, probs_index].flatten()
            p_line = P_hat[state][int(action)]
            if visited[state, int(action)]==0:
                P_hat[state][int(action)] = copy.copy(predicted_proba)
            else:
                P_hat[state][int(action)] = (p_line + copy.copy(predicted_proba)) / 2
            visited[state, int(action)] = 1.0

            P_diff.append(np.linalg.norm(P-P_hat))

            # Policy transition matrix P^pi
            for state in range(N_states):
                predicted_P_pi[state] = P_hat[state][policy[state]]
                true_P_pi[state] = P[state][policy[state]]
            P_pi_diff.append(np.linalg.norm(true_P_pi-predicted_P_pi))
            
            # KL divergence
            discrete_KL_loss.append(kl_div(p=true_proba,q=predicted_proba))
            
            # Bhattacharyya distance
            sqrt_PQ = np.sum(np.sqrt(predicted_proba * true_proba))
            BH = -np.log(sqrt_PQ)
            discrete_BT_loss.append(BH)

            # mode_prediction
            mode_arr.append(np.argmax(predicted_proba))

    series_dict['predicted_P'] = P_hat
            
    # loss_array = np.zeros(full_array[start_idx:].squeeze().shape)
    # # for state in states[[0,1]]:
    # for state in states:
    #     ### add 2 to slice out BOS
    #     pos = np.where(full_array[start_idx:]==state)[0]+2
    #     learned_p_out = probs.squeeze()[start_idx:][pos]
    #     true_p_out = torch.tensor(P[state])
    #     if dist_type == 'KL':
    #     ### KL divergence at each row
    #         KL = torch.sum(true_p_out * (true_p_out.log() - learned_p_out.log()), dim = 1)
    #         dist = KL
    #     else:
    #         ### Bhattacharyya distance at each row
    #         sqrt_PQ = torch.sum(torch.sqrt(learned_p_out * true_p_out), dim = 1)
    #         BH_dist = -torch.log(sqrt_PQ)
    #         dist = BH_dist
        
    #     loss_array[pos-2] = dist.numpy()

    statistics = {
        'mode_arr': np.array(mode_arr),
        'discrete_BT_loss': discrete_BT_loss,
        'discrete_KL_loss': discrete_KL_loss,
        'P_diff': np.array(P_diff),
        'P_pi_diff': np.array(P_pi_diff)
    }

    return series_dict, statistics

def plot_diff(series_dict, indices: List[int] = [0], add_actions: bool = False):
    series = series_dict['full_series'].split(',')
    actions = series_dict['actions']
    probs = series_dict['probs']
    P = series_dict['true_P']
    for index in indices:
        if (not add_actions) or (index%2 == 0):
            action = actions[index]
            if not np.isnan(action):
                state = int(series[index])
                true_proba = P[state][int(action)].flatten()
                predicted_proba = probs[0, 2*index if not add_actions else 2+4*i].flatten()
                plt.figure(figsize=(15,5))
                n=len(true_proba)
                r = np.arange(n) 
                width = 0.25
                plt.bar(r, true_proba, color = 'b', width = width, edgecolor = 'black',  label='gt')
                plt.bar(r + width, predicted_proba, color = 'orange', width = width, edgecolor = 'black', label='pred')
                plt.legend()
                plt.title(f"index: {index} with state: {state} and action: {action} | KL={kl_div(p=true_proba,q=predicted_proba)}")
                plt.show()
            else:
                print("terminal state")
        else:
            print("corresponds to action")