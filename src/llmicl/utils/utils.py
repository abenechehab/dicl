import numpy as np
import torch
from llmicl.utils.from_liu_et_al import MultiResolutionPDF


def calculate_multiPDF_llama3(
    full_series,
    model,
    tokenizer,
    n_states=1000,
    temperature=1.0,
    number_of_tokens_original=None,
    use_cache=False,
    kv_cache_prev=None,
):
    """
    This function calculates the multi-resolution probability density function (PDF) for a given series.

    Parameters:
    full_series (str): The series for which the PDF is to be calculated.
    prec (int): The precision of the PDF.
    mode (str, optional): The mode of calculation. Defaults to 'neighbor'.
    refine_depth (int, optional): The depth of refinement for the PDF. Defaults to 1.
    llama_size (str, optional): The size of the llama model. Defaults to '13b'.

    Returns:
    list: A list of PDFs for the series.
    """
    assert (
        n_states <= 1000
    ), f"if n_states ({n_states}) is larger than 1000, there will be more than 1 token per value!"

    good_tokens_str = []
    for num in range(n_states):
        good_tokens_str.append(str(num))
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    batch = tokenizer([full_series], return_tensors="pt", add_special_tokens=True)

    torch.cuda.empty_cache()
    with torch.no_grad():
        out = model(
            batch["input_ids"].cuda(),
            use_cache=use_cache,
            past_key_values=kv_cache_prev,
        )

    logit_mat = out["logits"]

    kv_cache_main = out["past_key_values"] if use_cache else None
    logit_mat_good = logit_mat[:, :, good_tokens].clone()

    if number_of_tokens_original:
        probs = torch.nn.functional.softmax(
            logit_mat_good[:, -(number_of_tokens_original - 1) :, :] / temperature,
            dim=-1,
        )
    else:
        probs = torch.nn.functional.softmax(
            logit_mat_good[:, 1:, :] / temperature, dim=-1
        )

    PDF_list = []

    # start_loop_from = 1 if use_instruct else 0
    for i in range(1, int(probs.shape[1]), 2):
        PDF = MultiResolutionPDF()
        PDF.bin_center_arr = np.arange(0, 1000) / 100
        PDF.bin_width_arr = np.array(1000 * [0.01])
        PDF.bin_height_arr = probs[0, i, :].cpu().numpy() * 100
        PDF_list.append(PDF)

    # release memory
    del logit_mat, kv_cache_prev  # , kv_cache_main
    return PDF_list, probs, kv_cache_main