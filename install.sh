conda create -n LLMICL python=3.9
conda activate LLMICL
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install openai 
pip install tiktoken
pip install tqdm
pip install matplotlib
pip install "pandas<2.0.0"
pip install transformers
pip install multiprocess
# conda deactivate LLMICL
