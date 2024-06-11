import numpy as np
import matplotlib.pyplot as plt
import torch

# colors = {1: 'dodgerblue', 0.1: 'violet', 0.01: 'hotpink'}
colors = {1: 'lightseagreen', 0.1: 'dodgerblue', 0.01: 'blue'}
# colors = {1: 'dodgerblue', 0.1: 'dodgerblue', 0.01: 'dodgerblue'}
def closest_color(width, colors):
    return colors[min(colors.keys(), key=lambda k: abs(k-width))]

def recursive_refiner(PDF, seq, curr = -3, refine_depth = -2, main = True, mode = "neighbor"
                ,model = None, tokenizer = None, good_tokens=None, kv_cache = None):
    """
    Recursively refines the PDF until desired depth

    Parameters:
        PDF (MultiResolutionPDF): The PDF to be refined.
        seq (str): must end with a number, not comma
        curr (int): The current precision. Default is -prec.
        refine_depth (int): The depth of refinement. Default is -2.
        main (bool): Whether the current sequence is on the main branch    
        kv_cache: cache of seq[0:-1]
        mode (str): "neighbor" or "all"
        model: transformer used for refinement.

    Returns:
    None
    """
    if curr == refine_depth:
        # print("nothing to refine, terminate refiner")
        return
    if main:
        main_digit = seq[curr]
        trimmed_seq = seq[:curr]
 
        if mode == "neighbor":
            # 2 off branches
            if curr < -1:
                trimmed_kv_cache = trim_kv_cache(kv_cache, curr+1)       
            for alt_digit in [int(main_digit) - 1, int(main_digit) + 1]:
                if alt_digit not in [10, -1]:
                    alt_seq = trimmed_seq + str(alt_digit)
                    recursive_refiner(PDF, alt_seq, curr, refine_depth, main = False, mode = "all"
                                      ,model = model, tokenizer = tokenizer, good_tokens=good_tokens, 
                                      kv_cache = trimmed_kv_cache) 
                    
        if mode == "all":
            # 9 off branches
            if curr < -1:
                trimmed_kv_cache = trim_kv_cache(kv_cache, curr+1)       
            for alt_digit in range(0, 10):
                if alt_digit != int(main_digit):
                    alt_seq = trimmed_seq + str(alt_digit)
                    recursive_refiner(PDF, alt_seq, curr, refine_depth, main = False, mode = "all",
                                      model = model, tokenizer = tokenizer, good_tokens=good_tokens, 
                                      kv_cache = trimmed_kv_cache)   
            
        if curr < refine_depth - 1:
            # skip to next main branch
            # no need to trim cache
            recursive_refiner(PDF, seq, curr+1, refine_depth, mode = "all",
                              model = model, tokenizer = tokenizer, main = True, good_tokens=good_tokens,
                              kv_cache = kv_cache)
    else:
        # ready to evaluate
        probs, kv_cache_new = next_token_prob_from_series(seq, kv_cache = kv_cache, model = model, tokenizer=tokenizer, good_tokens=good_tokens)
        last_comma_location = seq.rfind(',')
        num_slice = seq[last_comma_location+1:]
        last_digit_PDF = MultiResolutionPDF()
        last_digit_PDF.load_from_prec_digits_prob(num_slice, probs)
        PDF.refine(last_digit_PDF)
        if curr < refine_depth - 1:
            # 10 off branch
            for i in range(10):
                alt_seq = seq + str(i)
                recursive_refiner(PDF, alt_seq, curr+1, refine_depth, main = False, mode = "all",
                                  model = model, tokenizer = tokenizer, good_tokens=good_tokens,
                                  kv_cache = kv_cache_new) 
                
def trim_kv_cache(past_key_values, desired_length):
    """
    Trims the past_key_values cache along the sequence length dimension.
    Parameters:
        past_key_values (tuple): The original past_key_values cache, a nested tuple structure where
                                 each tuple corresponds to a layer in the transformer and contains
                                 two tensors: the key and value states.
        desired_length (int): The sequence length up to which you want to keep the cache.

    Returns:
        tuple: A new past_key_values cache where key and value states have been trimmed to the
               desired_length. The returned structure is a tuple of tuples.
    """    
    if past_key_values is None:
        return None
    trimmed_past_key_values = []
    for layer_past in past_key_values:
        # Each layer_past is a tuple (key_states, value_states)
        key_states, value_states = layer_past
        # Trim key_states and value_states along the sequence length dimension
        key_states = key_states[..., :desired_length, :]
        value_states = value_states[..., :desired_length, :]
        trimmed_past_key_values.append((key_states, value_states))
    return tuple(trimmed_past_key_values)

def next_token_prob_from_series(full_series, model = None, tokenizer = None, good_tokens = None, T=1, kv_cache = None, load_cache_to_cpu = False):
    """
    This function calculates the probability of the next token in a series.

    Parameters:
        full_series (str): The series of tokens.
        model (transformer): The transformer model to use for prediction.
        tokenizer (tokenizer): The tokenizer to use for tokenizing the series.
        T (int): Temperature parameter for softmax function. Default is 1.
        kv_cache (dict): The key-value cache for states [0:-1]

    Returns:
        tuple: A tuple containing the probabilities of the next token and the new key-value cache.
    """
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
    ### Put batch to cuda
    if kv_cache is None:
        with torch.no_grad():
            out = model(batch["input_ids"].cuda(), use_cache=True)
    else:
        if load_cache_to_cpu:
            kv_cache = tuple(tuple(x.cuda() for x in sub_tuple) for sub_tuple in kv_cache)
        with torch.no_grad():
            out = model(batch["input_ids"][:,-1:].cuda(), use_cache=True, past_key_values = kv_cache)

    logit_mat = out['logits'] 
    if load_cache_to_cpu:
        kv_cache_new = tuple(tuple(x.cpu() for x in sub_tuple) for sub_tuple in out['past_key_values'])
    else:
        kv_cache_new = out['past_key_values']
    probs = torch.nn.functional.softmax(logit_mat[0,-1,good_tokens].clone().cpu(), dim = 0).numpy()
    return (probs, kv_cache_new)


class MultiResolutionPDF:
    """
    A class for managing and visualizing probability density functions (PDFs)
    in a multi-resolution format.

    This class allows for adding data in the form of bins, normalizing the bins, 
    computing statistical properties (mean, mode, and standard deviation), plotting 
    the PDF, and evaluating the PDF at a given point.

    Attributes:
        bin_center_arr (numpy.array): Stores the centers of the bins.
        bin_width_arr (numpy.array): Stores the widths of the bins.
        bin_height_arr (numpy.array): Stores the heights of the bins.
        mode (float): The mode of the PDF, computed in `compute_stats`.
        mean (float): The mean of the PDF, computed in `compute_stats`.
        sigma (float): The standard deviation of the PDF, computed in `compute_stats`.
    """
    def __init__(self):
        """
        Constructor for the MultiResolutionPDF class.

        Initializes arrays for bin centers, widths, and heights. Statistical properties
        (mode, mean, sigma) are initialized to None.
        """
        self.bin_center_arr = np.array([])
        self.bin_width_arr = np.array([])
        self.bin_height_arr = np.array([])
        self.mode = None
        self.mean = None
        self.sigma = None

    def add_bin(self, center_arr, width_arr, height_arr, idx = None):
        """
        Adds bins to the PDF.
        Do not normalize because PDF may need multiple add_bin operations

        Args:
            center_arr (array_like): Array or list of bin centers.
            width_arr (array_like): Array or list of bin widths.
            height_arr (array_like): Array or list of bin heights.

        Raises:
            AssertionError: If the lengths of center_arr, width_arr, and height_arr are not equal.
        """
        # print(len(center_arr))
        # print(len(width_arr))
        # print(len(height_arr))
        assert len(center_arr) == len(width_arr) == len(height_arr), "center_arr, width_arr, height_arr must have the same length"
        if idx is None: # insert to index position
            self.bin_center_arr = np.append(self.bin_center_arr, center_arr)
            self.bin_width_arr = np.append(self.bin_width_arr, width_arr)
            self.bin_height_arr = np.append(self.bin_height_arr, height_arr)
        else: # append
            self.bin_center_arr = np.insert(self.bin_center_arr, idx, center_arr)
            self.bin_width_arr = np.insert(self.bin_width_arr, idx, width_arr)
            self.bin_height_arr = np.insert(self.bin_height_arr, idx, height_arr) 
        # self.normalize()                       
            
    def sort_by_center(self):
        """
        Sorts the bins by their centers.
        """
        if not np.all(np.diff(self.bin_center_arr) >= 0): # check if bin_center_arr is already sorted
            sort_indices = np.argsort(self.bin_center_arr) # sort by bin_center_arr
            self.bin_center_arr = self.bin_center_arr[sort_indices]
            self.bin_width_arr = self.bin_width_arr[sort_indices]
            self.bin_height_arr = self.bin_height_arr[sort_indices]
    
    def delete_by_idx(self, idx):
        """
        Deletes bins from the PDF by their indices.

        Args:
            idx (int or array_like): Index or list of indices of the bins to delete.
        """
        self.bin_center_arr = np.delete(self.bin_center_arr, idx)
        self.bin_width_arr = np.delete(self.bin_width_arr, idx)
        self.bin_height_arr = np.delete(self.bin_height_arr, idx) 
        
    def refine(self, Multi_PDF):
        """
        Refines the PDF by merging it with another MultiResolutionPDF.
        Reduce to add_bin if self empty

        Args:
            Multi_PDF (MultiResolutionPDF): Another MultiResolutionPDF to merge with.
        """
        if len(self.bin_center_arr) == 0:
            self.add_bin(Multi_PDF.bin_center_arr, Multi_PDF.bin_width_arr, Multi_PDF.bin_height_arr)
        else:    
            Multi_PDF.normalize()
            assert isinstance(Multi_PDF, MultiResolutionPDF), "Input must be an instance of MultiResolutionPDF"
            
            self.sort_by_center()
            right_edges = self.bin_center_arr + self.bin_width_arr/2
            insert_index = np.searchsorted(right_edges, Multi_PDF.bin_center_arr.min())
            insert_index_right = np.searchsorted(right_edges, Multi_PDF.bin_center_arr.max())
            # print(right_edges)
            # print(Multi_PDF.bin_center_arr)
            assert insert_index == insert_index_right, "refinement cannot straddle coarse bins"
            prefactor = self.bin_width_arr[insert_index] * self.bin_height_arr[insert_index] # probability of coase bin to replace
                    
            Multi_PDF.bin_height_arr *= prefactor
            self.delete_by_idx(insert_index)
            ### add bin, but to specific index
            self.add_bin(Multi_PDF.bin_center_arr, Multi_PDF.bin_width_arr, Multi_PDF.bin_height_arr, insert_index)

            # print(self.bin_center_arr)
            assert np.all(np.diff(self.bin_center_arr) >= 0), "final array should be sorted"
            self.check_gap_n_overlap()
            self.normalize()
            
    def coarsen(self, coarse_bin_centers, coarse_bin_widths):
        """
        Replace fine bins using coarse ones. This is for plotting purposes only.

        Args:
            coarse_bin_centers (np.ndarray): The centers of the coarse bins.
            coarse_bin_widths (np.ndarray): The widths of the coarse bins.
        """
        for coarse_bin_center, coarse_bin_width in zip(coarse_bin_centers, coarse_bin_widths):
            # Find the indices of the fine bins that fall within the coarse bin
            indices = np.where((self.bin_center_arr >= coarse_bin_center - coarse_bin_width / 2) &
                            (self.bin_center_arr <= coarse_bin_center + coarse_bin_width / 2))[0]

            if len(indices) == 0:
                continue

            # Compute the total height of the fine bins
            total_height = np.sum(self.bin_height_arr[indices] * self.bin_width_arr[indices])

            # Replace the fine bins with the coarse bin
            self.bin_center_arr = np.delete(self.bin_center_arr, indices)
            self.bin_width_arr = np.delete(self.bin_width_arr, indices)
            self.bin_height_arr = np.delete(self.bin_height_arr, indices)

            self.bin_center_arr = np.append(self.bin_center_arr, coarse_bin_center)
            self.bin_width_arr = np.append(self.bin_width_arr, coarse_bin_width)
            self.bin_height_arr = np.append(self.bin_height_arr, total_height / coarse_bin_width)

        # Sort the bins by their centers
        sort_indices = np.argsort(self.bin_center_arr)
        self.bin_center_arr = self.bin_center_arr[sort_indices]
        self.bin_width_arr = self.bin_width_arr[sort_indices]
        self.bin_height_arr = self.bin_height_arr[sort_indices]
        
    
    
    def load_from_num_prob(self, num_slice, prob_slice):
        """
        Loads the PDF from a given number slice and probability slice.

        Args:
            num_slice (array_like): The number slice to load from.
            prob_slice (array_like): The probability slice to load from.
        """
        assert len(num_slice) == len(prob_slice), "number of digits must equal number of probs"
        preceding_digits = None
        for idx, probs in enumerate(prob_slice):
            single_digit_PDF = MultiResolutionPDF()
            single_digit_PDF.load_from_prec_digits_prob(preceding_digits, probs)
            self.refine(single_digit_PDF)
            preceding_digits = num_slice[:idx+1]
            
    def load_from_prec_digits_prob(self, preceding_digits, probs):
        """
        Loads the PDF from a given preceding digits and probabilities of the last digit.

        Args:
            preceding_digits (array_like): The preceding digits, 
                which imply left_edge and bin_width
            probs (array_like): Distribution of next digit
        """
        assert len(probs.shape) == 1, "probs must be 1D"
        if preceding_digits is None:
            prec_len = 0
            w = 1
            left_edge = 0
        else:
            prec_len = len(preceding_digits)
            w = 0.1**prec_len
            left_edge = int(preceding_digits) * 10 * w
        x_coords = np.linspace(left_edge,left_edge+10 * w, 10, endpoint = False) + 0.5 * w
        
        self.add_bin(center_arr = x_coords,
                    width_arr = np.ones(10) * w,
                    height_arr = probs)
        self.normalize()    
    
    def normalize(self, report = False):
        """
        Normalizes the PDF so that the total area under the bins equals 1.
        Prints the total area before and after normalization.
        """
        total_area = np.sum(self.bin_width_arr * self.bin_height_arr)
        if total_area == 1.:
            if report:
                print('already normalized')
        else:
            if report:
                print('total area before normalization:', total_area)
            self.bin_height_arr = self.bin_height_arr / total_area
            
    def compute_stats(self):  
        """
        Computes and updates the statistical properties of the PDF: mean, mode, and standard deviation (sigma).
        """
        self.mean = np.sum(self.bin_center_arr * self.bin_width_arr * self.bin_height_arr)
        self.mode = self.bin_center_arr[np.argmax(self.bin_height_arr)]
        variance = np.sum(
            (self.bin_center_arr-self.mean) ** 2 * self.bin_height_arr * self.bin_width_arr
        )
        self.sigma = np.sqrt(variance)
            
    def compute_moment(self, n):
        """
        Computes the nth mean-centered moment of the PDF.

        Args:
            n (int): The order of the moment to compute.

        Returns:
            float: The nth moment of the PDF.
        """
        if self.mean is None:
            self.compute_stats()
        return np.sum(
            (self.bin_center_arr - self.mean) ** n * self.bin_height_arr * self.bin_width_arr
            )
        
    def rescale_temperature(self, alpha):
        """
        Rescale bins as if the original temperature 
        of softmax is scaled from T to alpha T
        """
        self.bin_height_arr = self.bin_height_arr ** (1/alpha)
        self.normalize()
        
    def check_gap_n_overlap(self):
        assert np.allclose(self.bin_center_arr[1:] - self.bin_width_arr[1:]/2, 
                           self.bin_center_arr[:-1] + self.bin_width_arr[:-1]/2), "bin overlap detected"
        
    def discretize(self, func, mode = 'pdf'):
        """
        Args:
            func: a function supported on self.bin_center_arr.
                  should be implmented using numpy operations for parallelization
            mode: 'pdf': approximate probability of bin using its center
                  'cdf': integrate over bin 
        Populate bin height by dicretizng
        """
        if mode == 'pdf':
            self.bin_height_arr = func(self.bin_center_arr)
        elif mode == 'cdf':
            right_edge = self.bin_center_arr + self.bin_width_arr/2
            left_edge = self.bin_center_arr - self.bin_width_arr/2
            prob_arr = func(right_edge) - func(left_edge)
            self.bin_height_arr = prob_arr / self.bin_width_arr
        self.normalize()
        
    def BT_dist(self, Multi_PDF):
        """
        Calculate the Bhattacharyya distance with another Multi_PDF object
        """          
        assert np.all(self.bin_center_arr == Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        weighted_PQ_arr = np.sqrt(self.bin_height_arr * Multi_PDF.bin_height_arr) * self.bin_width_arr
        return -np.log(np.sum(weighted_PQ_arr))
    
    def KL_div(self, Multi_PDF):
        """
        Calculate the KL divergence D_KL(self||Multi_PDF)
        Prone to numerical instabilities
        """          
        assert np.all(self.bin_center_arr == Multi_PDF.bin_center_arr), "Only PDFs of the same discretization are comparable"
        log_ratio = np.log(self.bin_height_arr) - np.log(Multi_PDF.bin_height_arr)
        weighted_log_ratio = log_ratio * self.bin_height_arr * self.bin_width_arr
        return np.sum(weighted_log_ratio)
        
    def plot(self, ax=None, log_scale=False, statistic = True):
        """
        Plots the PDF as a bar chart.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axis object. If None, a new figure and axis are created.
            log_scale (bool, optional): If True, sets the y-axis to logarithmic scale.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

        # ax.bar(self.bin_center_arr, self.bin_height_arr, width=self.bin_width_arr, align='center', color='black', alpha=0.5)
        # Define colors for different bin widths

        
        # Iterate over bins and plot with corresponding color
        for center, width, height in zip(self.bin_center_arr, self.bin_width_arr, self.bin_height_arr):
            color = closest_color(width, colors)
            ax.bar(center, height, width=width, align='center', color=color, alpha=1)
        
        
        if statistic:
            ax.vlines(self.mean, 0, np.max(self.bin_height_arr), color='blue', label='Mean', lw=2)
            ax.vlines(self.mode, 0, np.max(self.bin_height_arr), color='lightblue', label='Mode', lw=2)
            # Visualize sigma as horizontal lines
            ax.hlines(y=np.max(self.bin_height_arr), xmin=self.mean - self.sigma, xmax=self.mean + self.sigma, color='g', label='Sigma', lw=2)

        if log_scale:
            ax.set_yscale('log')

        ax.legend()

        # If ax was None, show the plot
        if ax is None:
            plt.show()

    def value_at(self, x):
        for center, width, height in zip(self.bin_center_arr, self.bin_width_arr, self.bin_height_arr):
            if center - width / 2 <= x <= center + width / 2:
                return height
        return 0
