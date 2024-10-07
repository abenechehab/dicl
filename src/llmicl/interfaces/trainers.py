from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional, List
from dataclasses import dataclass

import copy
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray

from scipy.special import erf

from llmicl.legacy.data.serialize import serialize_arr, SerializerSettings
from llmicl.matrix_completion.utils import (
    create_ns,
    completion_matrix,
    completion_matrix_ot_breg,
    bins_completion,
)
from llmicl.rl_helpers.rl_utils import (
    calculate_multiPDF_llama3,
    calculate_multiPDF,
    calculate_multiPDF_llama3_parallel,
)

from openai import OpenAI

if TYPE_CHECKING:
    from transformers import (
        LlamaForCausalLM,
        AutoTokenizer
    )
    from llmicl.legacy.models.ICL import MultiResolutionPDF


@dataclass
class ICLObject():
    time_series: Optional[NDArray[np.float32]] = None
    mean_series: Optional[NDArray[np.float32]] = None
    sigma_series: Optional[NDArray[np.float32]] = None
    str_series: Optional[str] = None
    rescaled_true_mean_arr: Optional[NDArray[np.float32]] = None
    rescaled_true_sigma_arr: Optional[NDArray[np.float32]] = None
    rescaling_min: Optional[NDArray[np.float32]] = None
    rescaling_max: Optional[NDArray[np.float32]] = None
    PDF_list: Optional[List] = None
    predictions: Optional[NDArray[np.float32]] = None
    mean_arr: Optional[NDArray[np.float32]] = None
    mode_arr: Optional[NDArray[np.float32]] = None
    sigma_arr: Optional[NDArray[np.float32]] = None
    moment_3_arr: Optional[NDArray[np.float32]] = None
    moment_4_arr: Optional[NDArray[np.float32]] = None
    kurtosis_arr: Optional[NDArray[np.float32]] = None
    kurtosis_error: Optional[NDArray[np.float32]] = None
    discrete_BT_loss: Optional[NDArray[np.float32]] = None
    discrete_KL_loss: Optional[NDArray[np.float32]] = None


class ICLTrainer(ABC):
    """ICLTrainer that takes a time serie and processes it using the LLM."""

    @abstractmethod
    def update_context(self, time_series: NDArray[np.float32], **kwargs) -> ICLObject:
        """Update the context (internal state) with the given time serie."""

    @abstractmethod
    def icl(self, **kwargs) -> ICLObject:
        """Calls the LLM and update the internal state with the PDF."""

    @abstractmethod
    def compute_statistics(self, **kwargs) -> ICLObject:
        """Compute useful statistics for the predicted PDFs in the internal state."""

    @abstractmethod
    def build_tranistion_matrices(self, **kwargs):
        """Build Markov chain transition kernels from the predicted PDFs and OT techniques."""

    @abstractmethod
    def predict_long_horizon_llm(self, **kwargs):
        """Long horizon autoregressive predictions using the LLM."""

    # @abstractstaticmethod
    # def predict_long_horizon_matrix(self, **kwargs):
    #     """Long horizon autoregressive predictions using the estimated transition kernel."""

    # @abstractmethod
    # def save(self, save_path: Path, **kwargs):
    #     """Save."""



class UnivariateICLTrainer(ICLTrainer):
    def __init__(
        self,
        model: "LlamaForCausalLM",
        tokenizer: "AutoTokenizer",
        rescale_factor: float = 7.0,
        up_shift: float = 1.5
    ):
        self.model: "LlamaForCausalLM" = model
        self.tokenizer: "AutoTokenizer" = tokenizer

        self.use_cache: bool = False

        self.up_shift: float = up_shift
        self.rescale_factor: float = rescale_factor

        self.icl_object: ICLObject = ICLObject()
        self.kv_cache: Optional[NDArray[np.float32]] = None

        self.transition_matrix_baseline = None
        self.transition_matrix_NN = None
        self.transition_matrix_OT = None

    def update_context(
        self,
        time_series: NDArray[np.float32],
        mean_series: NDArray[np.float32],
        sigma_series: NDArray[np.float32],
        context_length: int = 100,
        update_min_max: bool = True,
    ):
        self.context_length = context_length

        # ------------------ serialize_gaussian ------------------
        settings = SerializerSettings(
            base=10,
            prec=2,
            signed=True,
            time_sep=',',
            bit_sep='',
            minus_sign='-',
            fixed_length=False,
            max_val = 10
        )

        if update_min_max:
            self.icl_object.rescaling_min = time_series[: self.context_length].min()
            self.icl_object.rescaling_max = time_series[: self.context_length].max()

        ts_min = self.icl_object.rescaling_min
        ts_max = self.icl_object.rescaling_max
        rescaled_array = (time_series[: self.context_length] - ts_min) / (
            ts_max - ts_min
        ) * self.rescale_factor + self.up_shift
        rescaled_true_mean_arr = (mean_series[: self.context_length] - ts_min) / (
            ts_max - ts_min
        ) * self.rescale_factor + self.up_shift
        rescaled_true_sigma_arr = (
            sigma_series[: self.context_length]
            / (ts_max - ts_min)
            * self.rescale_factor
        )
        full_series = serialize_arr(rescaled_array, settings)

        self.icl_object.time_series = time_series[: self.context_length]
        self.icl_object.mean_series = mean_series[: self.context_length]
        self.icl_object.sigma_series = sigma_series[: self.context_length]
        self.icl_object.rescaled_true_mean_arr = rescaled_true_mean_arr
        self.icl_object.rescaled_true_sigma_arr = rescaled_true_sigma_arr
        self.icl_object.str_series = full_series

        return self.icl_object

    def icl(
        self,
        temperature: float = 1.0,
        n_states: int = 1000,
        use_cache: bool = False,
    ):
        self.use_cache = use_cache
        predictions = np.zeros((self.context_length))
        PDF_list, _, kv_cache = calculate_multiPDF_llama3(
            self.icl_object.str_series,
            model=self.model,
            tokenizer=self.tokenizer,
            n_states=n_states,
            temperature=temperature,
            use_cache=self.use_cache,
        )

        self.icl_object.PDF_list = PDF_list

        self.kv_cache = kv_cache

        ts_min = self.icl_object.rescaling_min
        ts_max = self.icl_object.rescaling_max

        for timestep in range(len(PDF_list)):
            PDF: "MultiResolutionPDF" = PDF_list[timestep]
            PDF.compute_stats()

            # Calculate the mode of the PDF
            next_state = (
                    (PDF.mode - self.up_shift) / self.rescale_factor
                ) * (ts_max - ts_min) + ts_min

            predictions[timestep] = next_state

        self.icl_object.predictions = predictions

        return self.icl_object

    def compute_statistics(self,):
        PDF_list: List["MultiResolutionPDF"] = self.icl_object.PDF_list
        PDF_true_list: List["MultiResolutionPDF"] = copy.deepcopy(PDF_list)

        ### Extract statistics from MultiResolutionPDF
        mean_arr = []
        mode_arr = []
        sigma_arr = []
        moment_3_arr = []
        moment_4_arr = []
        discrete_BT_loss = []
        discrete_KL_loss = []
        for PDF, PDF_true, true_mean, true_sigma in zip(
            PDF_list,
            PDF_true_list,
            self.icl_object.rescaled_true_mean_arr,
            self.icl_object.rescaled_true_sigma_arr,
        ):
            def cdf(x):
                return 0.5 * (1 + erf((x - true_mean) / (true_sigma * np.sqrt(2))))

            PDF_true.discretize(cdf, mode = "cdf")
            PDF_true.compute_stats()
            discrete_BT_loss.append([PDF_true.BT_dist(PDF)])
            discrete_KL_loss.append([PDF_true.KL_div(PDF)])

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

        self.icl_object.mean_arr = np.array(mean_arr)
        self.icl_object.mode_arr = np.array(mode_arr)
        self.icl_object.sigma_arr = np.array(sigma_arr)
        self.icl_object.moment_3_arr = np.array(moment_3_arr)
        self.icl_object.moment_4_arr = np.array(moment_4_arr)
        self.icl_object.kurtosis_arr = kurtosis_arr
        self.icl_object.kurtosis_error = (kurtosis_arr - 3) ** 2
        self.icl_object.discrete_BT_loss = np.array(discrete_BT_loss)
        self.icl_object.discrete_KL_loss = np.array(discrete_KL_loss)

        return self.icl_object

    def build_tranistion_matrices(self, reg: float = 5e-3, verbose: int = 0):
        comma_locations = np.sort(
            np.where(np.array(list(self.icl_object.str_series)) == ",")[0]
        )
        ns = create_ns(self.icl_object.str_series, comma_locations)
        bins_ = bins_completion(self.icl_object.PDF_list)

        p_ot, _ = completion_matrix_ot_breg(
            bins_, ns, self.icl_object.discrete_BT_loss, reg=reg, verbose=verbose
        )
        p_nn, _ = completion_matrix(bins_, ns, self.icl_object.discrete_BT_loss)

        self.transition_matrix_NN = p_nn
        self.transition_matrix_OT = p_ot
        return self.transition_matrix_NN, self.transition_matrix_OT

    def predict_long_horizon_llm(self, state: np.array, h: int, temperature: float = 1.0):
        # """
        # Predict h steps into the future by appending the previous prediction to the time series.
        # """
        # future_states = np.zeros((h, self.n_observations))
        # current_state = state.copy()

        # for step in tqdm(range(h), desc="prediction horizon"):
        #     next_state = self.predict_llm(current_state, temperature)
        #     future_states[step] = next_state
        #     current_state = next_state  # Use the predicted state as the new current state for the next step

        # return future_states
        return None



class RLICLTrainer(ICLTrainer):
    def __init__(
        self,
        model: Optional["LlamaForCausalLM"],
        tokenizer: "AutoTokenizer",
        n_observations: int,
        rescale_factor: float = 7.0,
        up_shift: float = 1.5,
    ):
        self.model: "LlamaForCausalLM" = model
        self.tokenizer: "AutoTokenizer" = tokenizer

        self.n_observations: int = n_observations

        self.use_cache: bool = False

        self.up_shift: float = up_shift
        self.rescale_factor: float = rescale_factor

        self.icl_object: List[ICLObject] = [
            ICLObject() for _ in range(self.n_observations)
        ]
        self.kv_cache: List[Optional[NDArray[np.float32]]] = [
            None for _ in range(self.n_observations)
        ]

        self.transition_matrix_baseline = None
        self.transition_matrix_NN: Optional[List[NDArray[np.float32]]] = []
        self.transition_matrix_OT: Optional[List[NDArray[np.float32]]] = []

    def update_context(
        self,
        time_series: NDArray[np.float32],
        mean_series: NDArray[np.float32],
        sigma_series: NDArray[np.float32],
        context_length: int = 100,
        update_min_max: bool = True,
    ):
        self.context_length = context_length
        assert len(time_series.shape) > 1 and time_series.shape[1]==self.n_observations, f"Not all observations are given in time series of shape: {time_series.shape}"

        for dim in range(self.n_observations):
            # ------------------ serialize_gaussian ------------------
            settings = SerializerSettings(base=10, prec=2, signed=True, time_sep=',', bit_sep='', minus_sign='-', fixed_length=False, max_val = 10)

            if update_min_max:
                self.icl_object[dim].rescaling_min = time_series[
                    : self.context_length, dim
                ].min()
                self.icl_object[dim].rescaling_max = time_series[
                    : self.context_length, dim
                ].max()

            ts_min = copy.copy(self.icl_object[dim].rescaling_min)
            ts_max = copy.copy(self.icl_object[dim].rescaling_max)
            rescaled_array = (time_series[: self.context_length, dim] - ts_min) / (
                ts_max - ts_min
            ) * self.rescale_factor + self.up_shift
            rescaled_true_mean_arr = (
                mean_series[: self.context_length, dim] - ts_min
            ) / (ts_max - ts_min) * self.rescale_factor + self.up_shift
            rescaled_true_sigma_arr = (
                sigma_series[: self.context_length, dim]
                / (ts_max - ts_min)
                * self.rescale_factor
            )

            full_series = serialize_arr(rescaled_array, settings)

            self.icl_object[dim].time_series = time_series[: self.context_length, dim]
            self.icl_object[dim].mean_series = mean_series[: self.context_length, dim]
            self.icl_object[dim].sigma_series = sigma_series[: self.context_length, dim]
            self.icl_object[dim].rescaled_true_mean_arr = rescaled_true_mean_arr
            self.icl_object[dim].rescaled_true_sigma_arr = rescaled_true_sigma_arr
            self.icl_object[dim].str_series = full_series
        return self.icl_object

    def icl(
        self,
        temperature: float = 1.0,
        n_states: int = 1000,
        stochastic: bool = False,
        use_cache: bool = False,
        verbose: int = 0,
        llama_3_tokenizer: bool = True,
        if_true_mean_else_mode: bool = False,
    ):
        self.use_cache = use_cache
        for dim in tqdm(
            range(self.n_observations),
            desc="icl / state dim",
            disable=not bool(verbose)
        ):
            if llama_3_tokenizer:
                PDF_list, _, kv_cache = calculate_multiPDF_llama3(
                    self.icl_object[dim].str_series,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    n_states=n_states,
                    temperature=temperature,
                    use_cache=self.use_cache,
                )
                self.kv_cache[dim] = kv_cache
            else:
                PDF_list = calculate_multiPDF(
                    self.icl_object[dim].str_series,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prec=2,
                )

            self.icl_object[dim].PDF_list = PDF_list

            ts_min = self.icl_object[dim].rescaling_min
            ts_max = self.icl_object[dim].rescaling_max

            predictions = []
            for timestep in range(len(PDF_list)):
                PDF: "MultiResolutionPDF" = PDF_list[timestep]
                PDF.compute_stats()

                # Calculate the mode of the PDF
                if stochastic:
                    raw_state = np.random.choice(
                        PDF.bin_center_arr,
                        p=PDF.bin_height_arr / np.sum(PDF.bin_height_arr),
                    )
                else:
                    raw_state = PDF.mean if if_true_mean_else_mode else PDF.mode
                next_state = ((raw_state - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min
                predictions.append(next_state)

            self.icl_object[dim].predictions = np.array(predictions)

        return self.icl_object

    def icl_parallel(
        self,
        temperature: float = 1.0,
        n_states: int = 1000,
        stochastic: bool = False,
        verbose: int = 0,
        if_true_mean_else_mode: bool = False,
    ):
        batched_PDF_list, _, _ = calculate_multiPDF_llama3_parallel(
            [self.icl_object[dim].str_series for dim in range(self.n_observations)],
            model=self.model,
            tokenizer=self.tokenizer,
            n_states=n_states,
            temperature=temperature,
        )

        for dim in tqdm(
            range(self.n_observations),
            desc="icl / state dim",
            disable=not bool(verbose)
        ):
            PDF_list = batched_PDF_list[dim]

            self.icl_object[dim].PDF_list = PDF_list

            ts_min = self.icl_object[dim].rescaling_min
            ts_max = self.icl_object[dim].rescaling_max

            predictions = []
            for timestep in range(len(PDF_list)):
                PDF: "MultiResolutionPDF" = PDF_list[timestep]
                PDF.compute_stats()

                # Calculate the mode of the PDF
                if stochastic:
                    raw_state = np.random.choice(
                        PDF.bin_center_arr,
                        p=PDF.bin_height_arr / np.sum(PDF.bin_height_arr),
                    )
                else:
                    raw_state = PDF.mean if if_true_mean_else_mode else PDF.mode
                next_state = ((raw_state - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min
                predictions.append(next_state)

            self.icl_object[dim].predictions = np.array(predictions)

        return self.icl_object

    def compute_statistics(self,):
        for dim in range(self.n_observations):
            PDF_list = self.icl_object[dim].PDF_list

            PDF_true_list = copy.deepcopy(PDF_list)

            ### Extract statistics from MultiResolutionPDF
            mean_arr = []
            mode_arr = []
            sigma_arr = []
            moment_3_arr = []
            moment_4_arr = []
            discrete_BT_loss = []
            discrete_KL_loss = []
            for PDF, PDF_true, true_mean, true_sigma in zip(
                PDF_list,
                PDF_true_list,
                self.icl_object[dim].rescaled_true_mean_arr,
                self.icl_object[dim].rescaled_true_sigma_arr,
            ):
                def cdf(x):
                    return 0.5 * (1 + erf((x - true_mean) / (true_sigma * np.sqrt(2))))
                PDF_true.discretize(cdf, mode = "cdf")
                PDF_true.compute_stats()
                discrete_BT_loss.append(PDF_true.BT_dist(PDF))
                discrete_KL_loss.append(PDF_true.KL_div(PDF))

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

            self.icl_object[dim].mean_arr = np.array(mean_arr)
            self.icl_object[dim].mode_arr = np.array(mode_arr)
            self.icl_object[dim].sigma_arr = np.array(sigma_arr)
            self.icl_object[dim].moment_3_arr = np.array(moment_3_arr)
            self.icl_object[dim].moment_4_arr = np.array(moment_4_arr)
            self.icl_object[dim].kurtosis_arr = kurtosis_arr
            self.icl_object[dim].kurtosis_error = (kurtosis_arr - 3) ** 2
            self.icl_object[dim].discrete_BT_loss = np.array(discrete_BT_loss)
            self.icl_object[dim].discrete_KL_loss = np.array(discrete_KL_loss)
        return self.icl_object

    def build_tranistion_matrices(self, reg: float = 5e-3, verbose: int = 0):
        self.transition_matrix_NN = []
        self.transition_matrix_OT = []
        for dim in tqdm(range(self.n_observations), desc="icl / state dim"):
            comma_locations = np.sort(
                np.where(np.array(list(self.icl_object[dim].str_series)) == ",")[0]
            )
            ns = create_ns(self.icl_object[dim].str_series, comma_locations)
            bins_ = bins_completion(self.icl_object[dim].PDF_list)

            p_ot, _ = completion_matrix_ot_breg(
                bins_,
                ns,
                self.icl_object[dim].discrete_BT_loss,
                reg=reg,
                verbose=verbose,
            )
            p_nn, _ = completion_matrix(
                bins_, ns, self.icl_object[dim].discrete_BT_loss
            )

            self.transition_matrix_NN.append(p_nn)
            self.transition_matrix_OT.append(p_ot)
        return self.transition_matrix_NN, self.transition_matrix_OT

    def predict_long_horizon_llm(
        self,
        prediction_horizon: int,
        temperature: float = 1.0,
        stochastic: bool = False,
        verbose: int = 0,
        if_true_mean_else_mode: bool = False,
    ):
        """
        Predict h steps into the future by appending the previous prediction to the time series.
        """
        last_prediction = copy.copy(np.concatenate(
            [
                self.icl_object[dim].predictions[-1].reshape((1, 1))
                for dim in range(self.n_observations)
            ],
            axis=1,
        ))

        current_ts = copy.copy(np.concatenate(
            [
                self.icl_object[dim].time_series.reshape((-1, 1))
                for dim in range(self.n_observations)
            ],
            axis=1,
        ))
        for h in tqdm(
            range(prediction_horizon),
            desc="prediction_horizon",
            disable=not bool(verbose),
        ):
            input_time_series = np.concatenate([current_ts, last_prediction], axis=0)

            self.update_context(
                time_series=input_time_series,
                mean_series=copy.copy(input_time_series),
                sigma_series=np.zeros_like(input_time_series),
                context_length=self.context_length + h + 1,
                update_min_max=False,  # if False, predictions get out of bounds
            )
            self.icl(
                temperature=temperature,
                stochastic=stochastic,
                if_true_mean_else_mode=if_true_mean_else_mode,
                verbose=0,
            )

            current_ts = np.concatenate([current_ts, last_prediction], axis=0)

            last_prediction = copy.copy(np.concatenate(
                [
                    self.icl_object[dim].predictions[-1].reshape((1, 1))
                    for dim in range(self.n_observations)
                ],
                axis=1,
            ))

        return self.compute_statistics()

    def predict_long_horizon_api(
        self,
        model: str = "llama-3-8b-instruct",
        prediction_horizon: int = 10,
        temperature: float = 1.0,
        verbose: int = 0,
    ):
        """
        Predict h steps into the future by appending the previous prediction to the time series.
        """
        all_predictions = []
        for dim in tqdm(
            range(self.n_observations),
            desc="icl / state dim",
            disable=not bool(verbose),
        ):
            # call openai api
            client = OpenAI(
                # base_url="http://10.155.97.225:4000/v1",
                base_url="http://10.227.91.60:4000/v1",  # For European Research Institue
                api_key="sk-1234",
            )

            good_tokens_str = [","]
            for num in range(1000):
                good_tokens_str.append(str(num))
            good_tokens = [
                self.tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str
            ]

            stream = client.chat.completions.create(
                model=model,
                messages=[
                    # {'role': 'system', 'content': 'You are a very helpful, respectful and honest assistant.'},
                    # {"role": "user", "content": "Tell me a joke."},
                    # {"role": "user", "content": chatgpt_sys_message + extra_input + fibo_series},
                    {"role": "user", "content": self.icl_object[dim].str_series},
                ],
                temperature=0,
                stream=True,
                max_tokens=2*prediction_horizon-1,
                logit_bias={id: 30 for id in good_tokens},
            )

            full_response = ''
            for chunk in stream:
                try:
                    full_response += chunk.choices[0].delta.content
                except TypeError:
                    pass
            full_response = full_response.split(',')

            # print(f"full_response: {full_response}")

            # self.icl_object[dim].PDF_list = PDF_list

            ts_min = self.icl_object[dim].rescaling_min
            ts_max = self.icl_object[dim].rescaling_max

            predictions = []
            for timestep in range(prediction_horizon):
                raw_state = float(full_response[timestep]) / 100
                next_state = ((raw_state - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min
                predictions.append(next_state)

            # self.icl_object[dim].predictions = np.array(predictions)

            all_predictions.append(predictions)

        return np.array(all_predictions)

    def predict_long_horizon_MC(
        self,
        prediction_horizon: int,
        sampling: str = 'mode',
    ):
        """
        Predict h steps into the future by rolling out the MC.
        """
        predictions = np.zeros(
            (self.context_length + prediction_horizon, self.n_observations)
        )

        # predictions for the time serie in context
        for dim in range(self.n_observations):
            ts_min = copy.copy(self.icl_object[dim].rescaling_min)
            ts_max = copy.copy(self.icl_object[dim].rescaling_max)
            for index, state in enumerate(self.icl_object[dim].str_series.split(',')[:-1]):
                next_state_dist = self.transition_matrix_OT[dim][int(state)]

                if sampling=='mode':
                    next_state = np.argmax(next_state_dist)
                elif sampling=='mean':
                    next_state = np.sum(next_state_dist * np.arange(1000))
                elif sampling=='stoch':
                    next_state = np.random.choice(
                        np.arange(1000),
                        p=next_state_dist / np.sum(next_state_dist),
                    )
                else:
                    raise ValueError(f'samplin "{sampling}" not supported!')

                predictions[index, dim] = (((next_state / 100) - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min

            for h in range(self.context_length, self.context_length + prediction_horizon):
                next_state_dist = self.transition_matrix_OT[dim][int(next_state)]

                if sampling=='mode':
                    next_state = np.argmax(next_state_dist)
                elif sampling=='mean':
                    next_state = np.sum(next_state_dist * np.arange(1000))
                elif sampling=='stoch':
                    next_state = np.random.choice(
                        np.arange(1000),
                        p=next_state_dist / np.sum(next_state_dist),
                    )
                else:
                    raise ValueError(f'samplin "{sampling}" not supported!')

                predictions[h, dim] = (((next_state / 100) - self.up_shift) / self.rescale_factor) * (
                    ts_max - ts_min
                ) + ts_min

        return predictions