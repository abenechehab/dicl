continuous_series_names = [
                           'brownian_motion', 
                           'geometric_brownian_motion',
                           'noisy_logistic_map',
                           'logistic_map',
                           'lorenz_system',
                           'uncorrelated_gaussian',
                           'correlated_gaussian',
                           'uncorrelated_uniform'
                           ]
markov_chain_names = ['markov_chain']


### Set up directory
import gc
import sys
import os
from pathlib import Path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Check if directory exists, if not create it
save_path = Path(parent_dir) / 'processed_series'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# Define the directory where the generated series are stored
generated_series_dir = Path(parent_dir) / 'generated_series'

import numpy as np

from tqdm import tqdm
import pickle
import torch
from llama import get_model_and_tokenizer
from ICL import MultiResolutionPDF, recursive_refiner, trim_kv_cache

def calculate_Markov(full_series, llama_size = '13b'):
    '''
     This function calculates the multi-resolution probability density function (PDF) for a given series.

     Parameters:
     full_series (str): The series for which the PDF is to be calculated.
     llama_size (str, optional): The size of the llama model. Defaults to '13b'.

     Returns:

    '''
    model, tokenizer = get_model_and_tokenizer(llama_size)
    states = sorted(set(full_series))
    good_tokens = [tokenizer.convert_tokens_to_ids(state) for state in states]
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True        
    )
    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(batch['input_ids'].cpu())
    logit_mat = out['logits']
    logit_mat_good = logit_mat[:,:,good_tokens].cpu()

    return logit_mat_good

model, tokenizer = get_model_and_tokenizer('13b')
def calculate_multiPDF(full_series, prec, mode = 'neighbor', refine_depth = 1, llama_size = '13b'):
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
    if llama_size != '13b':
        assert False, "Llama size must be '13b'"
    good_tokens_str = list("0123456789")
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    assert refine_depth < prec, "Refine depth must be less than precision"
    refine_depth = refine_depth - prec
    curr = -prec
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True        
    )
    torch.cuda.empty_cache()
    with torch.no_grad():
        # out = model(batch['input_ids'].cuda(), use_cache=True)
        out = model(batch['input_ids'].cpu(), use_cache=True)
    logit_mat = out['logits']
    kv_cache_main = out['past_key_values']
    logit_mat_good = logit_mat[:,:,good_tokens].clone()
    probs = torch.nn.functional.softmax(logit_mat_good[:,1:,:], dim=-1)
    
    PDF_list = []
    comma_locations = np.sort(np.where(np.array(list(full_series)) == ',')[0])

    for i in tqdm(range(len(comma_locations))):
        PDF = MultiResolutionPDF()
        # slice out the number before ith comma
        if i == 0:
            start_idx = 0
        else:
            start_idx = comma_locations[i-1]+1
        end_idx = comma_locations[i]
        num_slice = full_series[start_idx:end_idx]
        prob_slice = probs[0,start_idx:end_idx].cpu().numpy()
        ### Load hierarchical PDF 
        PDF.load_from_num_prob(num_slice, prob_slice)
        
        ### Refine hierarchical PDF
        seq = full_series[:end_idx]
        # cache and full_series are shifted from beginning, not end
        end_idx_neg = end_idx - len(full_series)
        ### kv cache contains seq[0:-1]
        kv_cache = trim_kv_cache(kv_cache_main, end_idx_neg-1)
        recursive_refiner(PDF, seq, curr = curr, main = True, refine_depth = refine_depth, mode = mode, 
                        kv_cache = kv_cache, model = model, tokenizer = tokenizer, good_tokens=good_tokens)

        PDF_list += [PDF]
        
    # release memory
    del logit_mat, kv_cache_main
    return PDF_list

# Initialize dictionaries to store the data for continuous series and Markov chains
continuous_series_task = {}
markov_chain_task = {}

# Loop through each file in the directory
for file in generated_series_dir.iterdir():
    # Check if a series is already processed
    if not (save_path / file.name).exists():\
        # Extract the series name from the file name
        series_name = file.stem.rsplit('_', 1)[0]
        # If the series is a continuous series, load the data into the continuous_series_data dictionary
        if series_name in continuous_series_names:
            continuous_series_task[file.name] = pickle.load(file.open('rb'))
        # If the series is a Markov chain, load the data into the markov_chain_data dictionary
        elif series_name in markov_chain_names:
            markov_chain_task[file.name] = pickle.load(file.open('rb'))
        # If the series name is not recognized, raise an exception
        # else:
        #     raise Exception(f"Unrecognized series name: {series_name}")
        
print(continuous_series_task.keys())
print(markov_chain_task.keys())

for series_name, series_dict in sorted(continuous_series_task.items()):
    prec = series_dict['prec']
    if prec != 20:
        print("Processing ", series_name)
        full_series = series_dict['full_series']
        prec = series_dict['prec']
        refine_depth = series_dict['refine_depth']
        llama_size = series_dict['llama_size']
        mode = series_dict['mode']
        PDF_list = calculate_multiPDF(full_series, prec, mode = mode, refine_depth = refine_depth, llama_size = llama_size)
        series_dict['PDF_list'] = PDF_list
        save_name = os.path.join(save_path, series_name)
        with open(save_name, 'wb') as f:
            pickle.dump(series_dict, f)
        # Clear memory
        del full_series, PDF_list, series_dict
        gc.collect()
        
        
for series_name, series_dict in sorted(markov_chain_task.items()):
    print("Processing ", series_name)
    full_series = series_dict['full_series']
    llama_size = series_dict['llama_size']
    logit_mat_good = calculate_Markov(full_series, llama_size = llama_size)    
    series_dict['logit_mat_good'] = logit_mat_good
    save_name = os.path.join(save_path, series_name)
    with open(save_name, 'wb') as f:
        pickle.dump(series_dict, f)