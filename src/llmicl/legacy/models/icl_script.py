

### Set up directory
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)



from llama import get_model_and_tokenizer


# --------------- load model ---------------
model, tokenizer = get_model_and_tokenizer('7b')