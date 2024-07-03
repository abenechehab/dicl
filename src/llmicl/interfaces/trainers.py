from ABC import abc

from typing import TYPE_CHECKING, NamedTuple, Optional, List

import copy
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from transformers import (
        LlamaForCausalLM, 
        AutoTokenizer
    )
    from gymnasium import Env


import torch




class ICLObject(NamedTuple):
    raw_time_serie: Optional[NDArray[np.float32]] = None
    str_series: Optional[str] = None
    rescaled_true_mean_arr: Optional[NDArray[np.float32]] = None
    rescaled_true_sigma_arr: Optional[NDArray[np.float32]] = None
    rescaling_min: Optional[NDArray[np.float32]] = None
    rescaling_max: Optional[NDArray[np.float32]] = None
    PDF_list: Optional[List] = None
    mean_arr: Optional[NDArray[np.float32]] = None
    mode_arr: Optional[NDArray[np.float32]] = None
    sigma_arr: Optional[NDArray[np.float32]] = None
    moment_3_arr: Optional[NDArray[np.float32]] = None
    moment_4_arr: Optional[NDArray[np.float32]] = None
    kurtosis_arr: Optional[NDArray[np.float32]] = None
    kurtosis_error: Optional[NDArray[np.float32]] = None

    


class ICLTrainer(abc):
    """ICLTrainer that takes a time serie and processes it using the LLM."""

    @abstractmethod
    def update_context(self, time_series, **kwargs):
        """Update the context (internal state) with the given time serie."""

    @abstractmethod
    def icl(self, **kwargs):
        """Calls the LLM and update the internal state with the PDF."""

    @abstractmethod
    def compute_statistics(self, **kwargs):
        """Compute useful statistics for the predicted PDFs in the internal state."""

    @abstractmethod
    def build_tranistion_matrices(self, **kwargs):
        """Build Markov chain transition kernels from the predicted PDFs and OT techniques."""

    @abstractstaticmethod
    def predict_long_horizon_llm(self, **kwargs):
        """Long horizon autoregressive predictions using the LLM."""

    # @abstractstaticmethod
    # def predict_long_horizon_matrix(self, **kwargs):
    #     """Long horizon autoregressive predictions using the estimated transition kernel."""

    # @abstractmethod
    # def save(self, save_path: Path, **kwargs):
    #     """Save."""



class UnivariateICLTrainer(ICLTrainer):
    def __init__(self, model: "LlamaForCausalLM", tokenizer: "AutoTokenizer", rescale_factor: float = 7.0, up_shift: float = 1.5):
        self.model = model
        self.tokenizer = tokenizer

        self.icl_object = ICLObject()  
        self.kv_cache = None

        self.transition_matrix_baseline = None
        self.transition_matrix_NN = None
        self.transition_matrix_OT = None
        
        self.use_cache = False

        self.up_shift = up_shift
        self.rescale_factor = rescale_factor 
        
    def update_context(self, time_series: np.array, context_length: int = 100, update_min_max: bool = True):
        self.context_length = context_length

        time_serie = time_series[:self.context_length].flatten()
        mean_series = copy.copy(time_serie)
        std_series = np.zeros_like(mean_series)
        
        # ------------------ serialize_gaussian ------------------
        settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=',', bit_sep='', minus_sign='-', fixed_length=False, max_val = 10)
        time_serie = np.array(time_serie)

        if update_min_max:
            self.rescaling_min = time_serie.min()
            self.rescaling_max = time_serie.max()
    
        rescaled_array = (time_serie-self.rescaling_min)/(self.rescaling_max-self.rescaling_min) * self.rescale_factor + self.up_shift
        rescaled_true_mean_arr = (np.array(mean_series)-self.rescaling_min)/(self.rescaling_max-self.rescaling_min) * self.rescale_factor + self.up_shift
        rescaled_true_sigma_arr = np.array(std_series)/(self.rescaling_max-self.rescaling_min) * self.rescale_factor
        
        full_series = serialize_arr(rescaled_array, settings)

        # Save the generated data to a dictionary
        series_dict = {
            'full_series': full_series,
            'rescaled_true_mean_arr': rescaled_true_mean_arr,
            'rescaled_true_sigma_arr': rescaled_true_sigma_arr,
            'time_series': time_serie,
        }
        self.in_context_series = series_dict
        return self.in_context_series
    
    def icl(self, temperature: float = 1.0, n_states: int = 1000, use_cache: bool = False, verbose: int = 0):
        self.use_cache = use_cache
        predictions = np.zeros((self.context_length, self.n_observations))
        for dim in tqdm(range(self.n_observations), desc="icl / state dim", disable=not bool(verbose)):
            PDF_list, probs, kv_cache = calculate_multiPDF_llama3(
                self.in_context_series[dim]['full_series'],
                model=self.model,
                tokenizer=self.tokenizer,
                n_states=n_states,
                temperature=temperature,
                use_cache=self.use_cache,
            )
            self.in_context_series[dim]['PDF_list'] = PDF_list
            self.in_context_series[dim]['probs'] = probs.detach().cpu().numpy()
            self.kv_cache[dim] = kv_cache

            ts_min = self.rescaling_min[dim]
            ts_max = self.rescaling_max[dim]
            
            for timestep in range(len(PDF_list)):
                PDF = PDF_list[timestep]
                PDF.compute_stats()
            
                # Calculate the mode of the PDF
                next_state = ((PDF.mode - self.up_shift) / self.rescale_factor) * (ts_max - ts_min) + ts_min
                predictions[timestep, dim] = next_state
        
        return predictions

    def compute_statistics(self,):
        for dim in range(self.n_observations):
            self.statistics[dim] = compute_statistics(
                series_dict=self.in_context_series[dim],
            )
        return self.statistics

    def build_tranistion_matrices(self, reg: float = 5e-3):
        for dim in range(self.n_observations):
            comma_locations = np.sort(np.where(np.array(list(self.in_context_series[dim]['full_series'])) == ',')[0])
            ns = create_ns(self.in_context_series[dim]['full_series'], comma_locations)
            bins_ = bins_completion(self.in_context_series[dim]['PDF_list'])

            p_ot, _ = completion_matrix_ot_breg(bins_,ns,statistics['discrete_BT_loss'], reg=reg)
            p_nn, _ = completion_matrix(bins_,ns,statistics['discrete_BT_loss'])

            self.transition_matrix_NN[dim] = p_nn
            self.transition_matrix_OT[dim] = p_ot
        return self.transition_matrix_NN, self.transition_matrix_OT

    def predict_long_horizon_llm(self, state: np.array, h: int, temperature: float = 1.0):
        """
        Predict h steps into the future by appending the previous prediction to the time series.
        """
        future_states = np.zeros((h, self.n_observations))
        current_state = state.copy()

        for step in tqdm(range(h), desc="prediction horizon"):
            next_state = self.predict_llm(current_state, temperature)
            future_states[step] = next_state
            current_state = next_state  # Use the predicted state as the new current state for the next step

        return future_states



class RLICLTrainer(ICLTrainer):
    def __init__(self, env: "Env", model: "LlamaForCausalLM", tokenizer: "AutoTokenizer", rescale_factor: float = 7.0, up_shift: float = 1.5):
        self.env = env
        self.model = model
        self.tokenizer = tokenizer

        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]

        self.rescaling_min = {}
        self.rescaling_max = {}

        self.in_context_series = {}
        self.statistics = {}
        self.kv_cache = {}

        self.transition_matrix_baseline = {}
        self.transition_matrix_NN = {}
        self.transition_matrix_OT = {}
        
        self.use_cache = False

        self.up_shift = up_shift
        self.rescale_factor = rescale_factor 
        
    def update_context(self, time_series: np.array, context_length: int = 100, update_min_max: bool = True):
        self.context_length = context_length
        for dim in range(self.n_observations):
            time_serie = time_series[:self.context_length,dim].flatten()
            # print(f"update_cntext | dim:{dim} | updated_series: {len(time_series)}")
            mean_series = copy.copy(time_serie)
            std_series = np.zeros_like(mean_series)
            
            # ------------------ serialize_gaussian ------------------
            settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=',', bit_sep='', minus_sign='-', fixed_length=False, max_val = 10)
            time_serie = np.array(time_serie)

            if update_min_max:
                self.rescaling_min[dim] = time_serie.min()
                self.rescaling_max[dim] = time_serie.max()
        
            rescaled_array = (time_serie-self.rescaling_min[dim])/(self.rescaling_max[dim]-self.rescaling_min[dim]) * self.rescale_factor + self.up_shift
            rescaled_true_mean_arr = (np.array(mean_series)-self.rescaling_min[dim])/(self.rescaling_max[dim]-self.rescaling_min[dim]) * self.rescale_factor + self.up_shift
            rescaled_true_sigma_arr = np.array(std_series)/(self.rescaling_max[dim]-self.rescaling_min[dim]) * self.rescale_factor
            
            full_series = serialize_arr(rescaled_array, settings)

            # Save the generated data to a dictionary
            series_dict = {
                'full_series': full_series,
                'rescaled_true_mean_arr': rescaled_true_mean_arr,
                'rescaled_true_sigma_arr': rescaled_true_sigma_arr,
                'time_series': time_serie,
            }
            self.in_context_series[dim] = series_dict
        return self.in_context_series
    
    def icl(self, temperature: float = 1.0, n_states: int = 1000, use_cache: bool = False, verbose: int = 0):
        self.use_cache = use_cache
        predictions = np.zeros((self.context_length, self.n_observations))
        for dim in tqdm(range(self.n_observations), desc="icl / state dim", disable=not bool(verbose)):
            PDF_list, probs, kv_cache = calculate_multiPDF_llama3(
                self.in_context_series[dim]['full_series'],
                model=self.model,
                tokenizer=self.tokenizer,
                n_states=n_states,
                temperature=temperature,
                use_cache=self.use_cache,
            )
            self.in_context_series[dim]['PDF_list'] = PDF_list
            self.in_context_series[dim]['probs'] = probs.detach().cpu().numpy()
            self.kv_cache[dim] = kv_cache

            ts_min = self.rescaling_min[dim]
            ts_max = self.rescaling_max[dim]
            
            for timestep in range(len(PDF_list)):
                PDF = PDF_list[timestep]
                PDF.compute_stats()
            
                # Calculate the mode of the PDF
                next_state = ((PDF.mode - self.up_shift) / self.rescale_factor) * (ts_max - ts_min) + ts_min
                predictions[timestep, dim] = next_state
        
        return predictions

    def compute_statistics(self,):
        for dim in range(self.n_observations):
            self.statistics[dim] = compute_statistics(
                series_dict=self.in_context_series[dim],
            )
        return self.statistics

    def build_tranistion_matrices(self, reg: float = 5e-3):
        for dim in range(self.n_observations):
            comma_locations = np.sort(np.where(np.array(list(self.in_context_series[dim]['full_series'])) == ',')[0])
            ns = create_ns(self.in_context_series[dim]['full_series'], comma_locations)
            bins_ = bins_completion(self.in_context_series[dim]['PDF_list'])

            p_ot, _ = completion_matrix_ot_breg(bins_,ns,statistics['discrete_BT_loss'], reg=reg)
            p_nn, _ = completion_matrix(bins_,ns,statistics['discrete_BT_loss'])

            self.transition_matrix_NN[dim] = p_nn
            self.transition_matrix_OT[dim] = p_ot
        return self.transition_matrix_NN, self.transition_matrix_OT

    def predict_long_horizon_llm(self, state: np.array, h: int, temperature: float = 1.0):
        """
        Predict h steps into the future by appending the previous prediction to the time series.
        """
        future_states = np.zeros((h, self.n_observations))
        current_state = state.copy()

        for step in tqdm(range(h), desc="prediction horizon"):
            next_state = self.predict_llm(current_state, temperature)
            future_states[step] = next_state
            current_state = next_state  # Use the predicted state as the new current state for the next step

        return future_states