import torch
import numpy as np
from jax import grad,vmap
from tqdm.notebook import tqdm

from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoTokenizer
)
from data.serialize import serialize_arr, SerializerSettings

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

def get_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]  
    
    # tokenizer = LlamaTokenizer.from_pretrained(
    # "/data01/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/",
    # use_fast=False,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
    "/data01/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45/",
    use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer  

def get_model_and_tokenizer(model):
    name_parts = model.split("-")
    model_size = name_parts[0]
    chat = len(name_parts) > 1

    assert model_size in ["7b", "13b", "70b"]
    print(f"getting tokenizer")
    tokenizer = get_tokenizer(model)
    print(f"getting model")
    model = LlamaForCausalLM.from_pretrained(
        # "/data01/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/",
        "/home/abenechehab/hugguing_face_hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/",
        device_map="auto",   
        torch_dtype=torch.float16,
    )
    print(f"finish loading model")
    # model = LlamaForCausalLM.from_pretrained(
    # "/home/admin-quad/LLM/llama/llama-2-7b/",
    # device_map="auto",   
    # torch_dtype=torch.float16,
    # )
    
    model.eval()

    return model, tokenizer

def tokenize_fn(str, model):
    tokenizer = get_tokenizer(model)
    return tokenizer(str)

def llama_nll_fn(model, input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1):
    """ Returns the NLL/dimension (log base e) of the target array (continuous) according to the LM 
        conditioned on the input array. Applies relevant log determinant for transforms and
        converts from discrete NLL of the LLM to continuous by assuming uniform within the bins.
    inputs:
        input_arr: (n,) context array
        target_arr: (n,) ground truth array
    Returns: NLL/D
    """
    model, tokenizer = get_model_and_tokenizer(model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    
    print("train_str: ...", input_str[-20:])
    print("test_str: ...", target_str[:20])
    full_series = input_str + target_str
    
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out['logits'][:,:,bad_tokens] = -100
    # why not set to -inf?

    input_ids = batch['input_ids'][0][1:]
    logprobs = torch.nn.functional.log_softmax(out['logits'], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()

    tokens = tokenizer.batch_decode(
        input_ids,
        skip_special_tokens=False, 
        clean_up_tokenization_spaces=False
    )
    
    input_len = len(tokenizer([input_str], return_tensors="pt",)['input_ids'][0])
    input_len = input_len - 2 # remove the BOS token

    logprobs = logprobs[input_len:]
    tokens = tokens[input_len:]
    BPD = -logprobs.sum()/len(target_arr)

    #print("BPD unadjusted:", -logprobs.sum()/len(target_arr), "BPD adjusted:", BPD)
    # log p(x) = log p(token) - log bin_width = log p(token) + prec * log base
    transformed_nll = BPD - settings.prec*np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    return transformed_nll-avg_logdet_dydx

def llama_completion_fn(
    model,
    input_str,
    steps, # len(test[0])
    settings,
    batch_size=1,
    num_samples=20,
    temp=0.9, 
    top_p=0.9,
):
    input_len = len(input_str.split(settings.time_sep))
    avg_tokens_per_step = len(input_str)/input_len
    max_tokens = int(avg_tokens_per_step*steps)
    
    model, tokenizer = get_model_and_tokenizer(model)
    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(str) for str in good_tokens_str]
    good_tokens += [tokenizer.eos_token_id]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    
    batch = tokenizer([input_str], return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    
    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp, 
            top_p=top_p, 
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

    gen_strs = [x.replace(input_str, '').strip() for x in gen_strs]

    return gen_strs
